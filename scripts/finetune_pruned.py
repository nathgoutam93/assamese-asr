"""Fine-tune a layer-pruned Assamese CTC model on IndicVoices-as (shrinking plan A, step 3).

Model: the hybrid checkpoint's encoder with `--drop` layers removed + the Assamese
slice of the CTC head (as in scripts/export_onnx.py). Loss: CTC on the
transcripts, optionally plus frame-level KL to the unpruned model's CTC
posteriors (`--kd-weight`; teacher sees clean features, student sees SpecAugment).
Single GPU, plain PyTorch loop: the RNNT branch and the 22-language head are
not trained, so NeMo's hybrid training setup is not needed.

Outputs in --out: best.pt (lowest dev WER), last.pt (resumable; restarted with
--resume), log.jsonl. Evaluate with scripts/eval_checkpoint.py, export with
`scripts/export_onnx.py --checkpoint`.

Usage (see docs/finetune-gpu.md):
  python scripts/finetune_pruned.py --drop 2,3,8,9 --out runs/drop4
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from itertools import count
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from indic_asr.pruning import (
    LANGUAGE_ID,
    batches,
    clean_target,
    count_params,
    ctc_greedy,
    decode_audio,
    encode_batch,
    iv_shards,
    keep_layers,
    load_ctc_model,
    load_indicvoices,
    save_checkpoint,
    score,
    target_chars,
)

FRAMES_PER_SEC = 25  # 10 ms mel hop, 4x subsampling


# --------------------------------------------------------------------------- data


class IVStream(IterableDataset):
    """Shuffled stream of (audio, token ids) from IndicVoices-as parquet shards.

    Unit of work is a parquet row group (~740 utterances): each epoch the units
    are shuffled and dealt to DataLoader workers, rows are shuffled within a
    unit, audio is decoded in the worker. Set `.epoch` before iterating.
    """

    def __init__(self, files, sp, allowed, min_dur, max_dur, seed):
        import pyarrow.parquet as pq

        self.units = [(f, g) for f in files for g in range(pq.ParquetFile(f).num_row_groups)]
        self.sp, self.allowed = sp, allowed
        self.min_dur, self.max_dur, self.seed = min_dur, max_dur, seed
        self.epoch = 0

    def __iter__(self):
        import pyarrow.parquet as pq

        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        units = list(self.units)
        random.Random(self.seed + self.epoch).shuffle(units)
        rng = random.Random(self.seed * 1000 + self.epoch * 100 + wid)
        for f, g in units[wid::nw]:
            rows = (
                pq.ParquetFile(f)
                .read_row_group(g, columns=["audio_filepath", "text", "duration"])
                .to_pylist()
            )
            rng.shuffle(rows)
            for r in rows:
                if not self.min_dur <= r["duration"] <= self.max_dur:
                    continue
                text = clean_target(r["text"], self.allowed)
                if text is None:
                    continue
                ids = self.sp.text_to_ids(text)
                # CTC needs at least one frame per token (more with repeats)
                if not ids or len(ids) > r["duration"] * FRAMES_PER_SEC * 0.9:
                    continue
                yield decode_audio(r["audio_filepath"]["bytes"]), ids


def train_hours(files, min_dur, max_dur) -> float:
    """Audio hours within the duration limits (reads only the duration column)."""
    import pyarrow.parquet as pq

    d = np.concatenate([pq.read_table(f, columns=["duration"])["duration"].to_numpy() for f in files])
    return float(d[(d >= min_dur) & (d <= max_dur)].sum() / 3600)


def bucketed(samples, batch_seconds, buffer_size, rng):
    """Group a sample stream into duration-sorted batches of <= batch_seconds
    padded audio, shuffled within a buffer so batch order stays random."""

    def flush(buf):
        buf.sort(key=lambda s: len(s[0]))
        out, cur = [], []
        for s in buf:
            if cur and len(s[0]) / 16_000 * (len(cur) + 1) > batch_seconds:
                out.append(cur)
                cur = []
            cur.append(s)
        if cur:
            out.append(cur)
        rng.shuffle(out)
        return out

    buf = []
    for s in samples:
        buf.append(s)
        if len(buf) >= buffer_size:
            yield from flush(buf)
            buf = []
    if buf:
        yield from flush(buf)


def _identity(x):
    return x  # keep numpy audio as-is; collate() pads in the main process


def collate(batch, device):
    lens = torch.tensor([len(a) for a, _ in batch])
    x = torch.zeros(len(batch), int(lens.max()))
    for i, (a, _) in enumerate(batch):
        x[i, : len(a)] = torch.from_numpy(a)
    tlens = torch.tensor([len(t) for _, t in batch])
    targets = torch.tensor([i for _, t in batch for i in t])
    return (
        x.to(device, non_blocking=True),
        lens.to(device),
        targets.to(device),
        tlens.to(device),
    )


# --------------------------------------------------------------------------- train


def lr_lambda(warmup, total, min_ratio):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    return f


@torch.no_grad()
def evaluate(model, utts):
    """fp32, same code path as prune_layers.py, so numbers are directly comparable."""
    model.eval()
    refs, hyps = [], []
    for b in batches(utts):
        enc, elen, _ = encode_batch(model, b)
        hyps += ctc_greedy(model, enc, elen)
        refs += [u.text for u in b]
    model.train()
    return score(refs, hyps)


def masked_kl(t_logp, s_logp, lens):
    """sum over valid frames of KL(teacher || student), averaged over the batch
    (same normalisation as the CTC loss below)."""
    kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)  # [B, T]
    mask = torch.arange(kl.shape[1], device=kl.device)[None] < lens[:, None]
    return (kl * mask).sum() / kl.shape[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drop", required=True, help="encoder layers to remove, e.g. 2,3,8,9 (0-based, of 17)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=30_000)
    ap.add_argument("--lr", type=float, default=1e-4, help="peak LR (original training peaked ~4e-4)")
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch-seconds", type=float, default=600, help="padded audio per batch")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--kd-weight", type=float, default=0.0, help="weight of KL to the unpruned model (try 1.0)")
    ap.add_argument("--min-duration", type=float, default=0.5)
    ap.add_argument("--max-duration", type=float, default=20.0)
    ap.add_argument("--max-shards", type=int, default=0, help="use only the first N train shards (0 = all 112)")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--dev-n", type=int, default=400, help="dev utterances (400/seed 0 = prune_layers.py set)")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--buffer", type=int, default=4000, help="samples per bucketing buffer")
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    args = ap.parse_args()
    if (args.out / "last.pt").exists() and not args.resume:
        raise SystemExit(f"{args.out / 'last.pt'} exists: pass --resume to continue it, or use a new --out")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]
    if dev.type != "cuda":
        amp_dtype = None

    # ---- model
    model = load_ctc_model(args.device)
    n_layers = len(model.encoder.layers)
    drop = sorted({int(i) for i in args.drop.split(",") if i.strip()})
    assert all(0 <= i < n_layers for i in drop), f"--drop must be in 0..{n_layers - 1}"
    keep = [i for i in range(n_layers) if i not in drop]
    keep_layers(model, keep)
    params = list(model.encoder.parameters()) + list(model.ctc_decoder.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    print(f"student: {len(keep)} layers, {count_params(model) / 1e6:.1f}M params, dropped {drop}", flush=True)

    teacher = None
    if args.kd_weight > 0:
        teacher = load_ctc_model(args.device)
        teacher.requires_grad_(False)

    sp = model.tokenizer.tokenizers_dict[LANGUAGE_ID]
    blank = sp.vocab_size  # sliced head: Assamese tokens 0..255, blank last

    # ---- data
    print("fetching train shards...", flush=True)
    files = iv_shards(args.train_split, args.max_shards or None)
    ds = IVStream(files, sp, target_chars(model), args.min_duration, args.max_duration, args.seed)
    hours = train_hours(files, args.min_duration, args.max_duration)
    print(
        f"train: {len(files)} shards, {len(ds.units)} row groups, {hours:.0f} h in duration range "
        f"(text filtering drops a few more); ~{hours * 3600 / args.batch_seconds / args.accum:.0f} "
        "steps/epoch at full batches",
        flush=True,
    )
    dev_utts = load_indicvoices("valid", n=args.dev_n or None, seed=0)
    print(f"dev: {len(dev_utts)} utterances, {sum(u.duration for u in dev_utts) / 3600:.2f} h", flush=True)

    # ---- optim
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(args.warmup, args.max_steps, args.min_lr_ratio))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    step, epoch, best = 0, 0, float("inf")
    last_path, best_path = args.out / "last.pt", args.out / "best.pt"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=dev, weights_only=True)
        assert ck["keep"] == keep, f"checkpoint has layers {ck['keep']}, --drop gives {keep}"
        model.encoder.load_state_dict(ck["encoder"])
        model.ctc_decoder.load_state_dict(ck["ctc_decoder"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        step, epoch, best = ck["step"], ck["epoch"] + 1, ck["best_wer"]
        print(f"resumed at step {step} (epoch {epoch}, best dev WER {best:.2f})", flush=True)

    log = open(args.out / "log.jsonl", "a")

    def emit(rec):
        rec = {"step": step, "time": time.time(), **rec}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    def checkpoint():
        nonlocal best
        m = evaluate(model, dev_utts)
        if m["wer"] < best:
            best = m["wer"]
            save_checkpoint(best_path, model, keep, step=step, dev=m, args=vars(args) | {"out": str(args.out)})
        save_checkpoint(
            last_path, model, keep, step=step, epoch=epoch, best_wer=best,
            opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
        )
        emit({"dev_wer": m["wer"], "dev_cer": m["cer"], "best_wer": best})

    if step == 0:
        checkpoint()  # WER of the pruned model before any training

    # ---- loop
    model.train()
    if teacher is not None:
        teacher.eval()
    bucket_rng = random.Random(args.seed)
    t0, seen_sec, run_loss, run_kd, n_run = time.time(), 0.0, 0.0, 0.0, 0
    micro = 0
    for epoch in count(epoch):
        ds.epoch = epoch
        loader = DataLoader(
            ds, batch_size=None, collate_fn=_identity, num_workers=args.workers,
            prefetch_factor=64 if args.workers else None,
        )
        for batch in bucketed(loader, args.batch_seconds, args.buffer, bucket_rng):
            x, lens, targets, tlens = collate(batch, dev)
            with torch.no_grad():
                feats, flen = model.preprocessor(input_signal=x, length=lens)
            aug = model.spec_augmentation(input_spec=feats.clone(), length=flen)
            with torch.autocast(dev.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                enc, elen = model.encoder(audio_signal=aug, length=flen)
                logp = model.ctc_decoder(encoder_output=enc)
            logp = logp.float()
            ctc = F.ctc_loss(
                logp.transpose(0, 1), targets, elen, tlens, blank=blank, reduction="sum", zero_infinity=True
            ) / len(batch)
            loss = ctc
            if teacher is not None:
                with torch.no_grad(), torch.autocast(
                    dev.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None
                ):
                    t_enc, _ = teacher.encoder(audio_signal=feats, length=flen)
                    t_logp = teacher.ctc_decoder(encoder_output=t_enc).float()
                kd = masked_kl(t_logp, logp, elen)
                loss = ctc + args.kd_weight * kd
                run_kd += kd.item()

            scaler.scale(loss / args.accum).backward()
            run_loss += ctc.item()
            n_run += 1
            seen_sec += lens.sum().item() / 16_000
            micro += 1
            if micro % args.accum:
                continue

            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % args.log_every == 0:
                dt = time.time() - t0
                rec = {
                    "epoch": epoch, "ctc": run_loss / n_run, "lr": sched.get_last_lr()[0],
                    "gnorm": float(gnorm), "audio_h_per_h": seen_sec / dt, "batch": len(batch),
                }
                if teacher is not None:
                    rec["kd"] = run_kd / n_run
                emit(rec)
                t0, seen_sec, run_loss, run_kd, n_run = time.time(), 0.0, 0.0, 0.0, 0
            if step % args.eval_every == 0 or step >= args.max_steps:
                checkpoint()
            if step >= args.max_steps:
                print(f"done: best dev WER {best:.2f} -> {best_path}", flush=True)
                return
        emit({"epoch_done": epoch})


if __name__ == "__main__":
    main()
