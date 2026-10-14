"""Layer-pruning sweep, no training (docs/shrinking-plan.md, experiment A steps 1-2).

Scores the Assamese CTC model on a fixed IndicVoices-as subset with:
  baseline       all 17 layers
  top-k          first k layers only, every k (one pass, via per-layer outputs)
  skip-i         every layer except i, each i (per-layer importance)
  drop-N/middle  N contiguous layers removed from the middle
  drop-N/least   the N layers whose single removal hurt least

Results go to <out>/results.json (resumable: finished configs are skipped) and
<out>/results.md.

Usage: python scripts/prune_layers.py [--n 400] [--seed 0] [--out experiments/layer_pruning]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from assamese_asr.pruning import (
    batches,
    count_params,
    ctc_greedy,
    encode_batch,
    layers_kept,
    load_ctc_model,
    load_indicvoices,
    score,
)

DROP_COUNTS = (4, 6, 8)


def run(model, utts, keep=None):
    """Score with only `keep` layers (all if None). Returns metrics + params."""
    n_layers = len(model.encoder.layers)
    keep = list(range(n_layers)) if keep is None else list(keep)
    refs, hyps = [], []
    t0 = time.time()
    with layers_kept(model, keep):
        params = count_params(model)
        for b in batches(utts):
            enc, elen, _ = encode_batch(model, b)
            hyps += ctc_greedy(model, enc, elen)
            refs += [u.text for u in b]
    return {"keep": keep, "params_m": params / 1e6, "secs": time.time() - t0, **score(refs, hyps)}


def run_top_k(model, utts):
    """Every 'first k layers' cut from a single full pass."""
    n_layers = len(model.encoder.layers)
    refs, hyps = [], {k: [] for k in range(1, n_layers + 1)}
    for b in batches(utts):
        _, elen, outs = encode_batch(model, b, capture_layers=True)
        refs += [u.text for u in b]
        for k in hyps:
            hyps[k] += ctc_greedy(model, outs[k - 1], elen)
    res = {}
    for k, h in hyps.items():
        with layers_kept(model, range(k)):
            params = count_params(model)
        res[f"top-{k}"] = {"keep": list(range(k)), "params_m": params / 1e6, **score(refs, h)}
    return res


def write_md(path: Path, results: dict, meta: dict) -> None:
    base = results["baseline"]
    header = ["| config | layers | params (M) | WER | CER | ΔWER |", "|---|---|---|---|---|---|"]
    lines = [
        "# Layer pruning sweep (no fine-tuning)",
        "",
        f"Eval: IndicVoices-as `{meta['split']}`, {meta['n_utts']} utterances "
        f"({meta['hours']:.2f} h), seed {meta['seed']}. Greedy CTC, Assamese head. "
        "WER/CER after removing punctuation.",
        "",
        *header,
    ]

    def row(name, r):
        return (
            f"| {name} | {len(r['keep'])} | {r['params_m']:.1f} | {r['wer']:.2f} | {r['cer']:.2f} "
            f"| {r['wer'] - base['wer']:+.2f} |"
        )

    lines.append(row("baseline", base))
    lines += ["", "**Drop N layers**", "", *header]
    for n in DROP_COUNTS:
        for strat in ("top", "middle", "least"):
            key = f"top-{17 - n}" if strat == "top" else f"drop-{n}/{strat}"
            if key in results:
                dropped = sorted(set(range(17)) - set(results[key]["keep"]))
                lines.append(row(f"drop-{n}/{strat} {dropped}", results[key]))
    lines += ["", "**Single-layer importance (skip one layer)**", "", *header]
    for i in range(17):
        if f"skip-{i}" in results:
            lines.append(row(f"skip-{i}", results[f"skip-{i}"]))
    lines += ["", "**Keep first k layers**", "", *header]
    for k in range(1, 18):
        if f"top-{k}" in results:
            lines.append(row(f"top-{k}", results[f"top-{k}"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="utterances to sample (0 = whole split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    ap.add_argument("--out", type=Path, default=Path("experiments/layer_pruning"))
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    res_path = args.out / "results.json"

    utts = load_indicvoices(args.split, n=args.n or None, seed=args.seed)
    meta = {
        "split": args.split,
        "seed": args.seed,
        "n_utts": len(utts),
        "hours": sum(u.duration for u in utts) / 3600,
    }
    saved = json.loads(res_path.read_text()) if res_path.exists() else {}
    if saved.get("meta") != meta:
        saved = {"meta": meta, "results": {}}  # different eval set: start over
    results = saved["results"]
    print(f"eval set: {meta}", flush=True)

    model = load_ctc_model()
    n_layers = len(model.encoder.layers)

    def save():
        res_path.write_text(json.dumps(saved, indent=1, ensure_ascii=False))
        write_md(args.out / "results.md", results, meta)

    def do(name, keep=None):
        if name in results:
            return
        results[name] = run(model, utts, keep)
        r = results[name]
        print(f"{name:16s} layers={len(r['keep']):2d} WER={r['wer']:6.2f} CER={r['cer']:6.2f} "
              f"({r['secs']:.0f}s)", flush=True)
        save()

    do("baseline")
    if "top-1" not in results:
        results.update(run_top_k(model, utts))
        save()
        print("top-k done", flush=True)

    for i in range(n_layers):
        do(f"skip-{i}", [j for j in range(n_layers) if j != i])

    importance = sorted(range(n_layers), key=lambda i: (results[f"skip-{i}"]["wer"], results[f"skip-{i}"]["cer"]))
    for n in DROP_COUNTS:
        start = (n_layers - n) // 2
        do(f"drop-{n}/middle", [j for j in range(n_layers) if not start <= j < start + n])
        least = set(importance[:n])
        do(f"drop-{n}/least", [j for j in range(n_layers) if j not in least])

    print((args.out / "results.md").read_text())


if __name__ == "__main__":
    main()
