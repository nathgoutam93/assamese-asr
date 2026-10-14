"""WER/CER of the base model or a pruned checkpoint on IndicVoices-as.

Usage:
  python scripts/eval_checkpoint.py                                # unpruned baseline
  python scripts/eval_checkpoint.py --checkpoint runs/drop4/best.pt
  python scripts/eval_checkpoint.py --checkpoint ... --n 0         # whole valid split
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from assamese_asr.pruning import batches, count_params, ctc_greedy, encode_batch, load_indicvoices, load_pruned_model, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="pruned checkpoint from finetune_pruned.py (default: base model)")
    ap.add_argument("--n", type=int, default=400, help="utterances (0 = whole split; 400/seed 0 = sweep set)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--show", type=int, default=5, help="print this many example hypotheses")
    args = ap.parse_args()

    model = load_pruned_model(args.checkpoint, args.device)
    utts = load_indicvoices(args.split, n=args.n or None, seed=args.seed)
    refs, hyps = [], []
    t0 = time.time()
    for b in batches(utts):
        enc, elen, _ = encode_batch(model, b)
        hyps += ctc_greedy(model, enc, elen)
        refs += [u.text for u in b]
    for r, h in list(zip(refs, hyps))[: args.show]:
        print(f"ref: {r}\nhyp: {h}\n")
    res = {
        "checkpoint": args.checkpoint or "base",
        "layers": len(model.encoder.layers),
        "params_m": round(count_params(model) / 1e6, 1),
        "split": args.split,
        "n_utts": len(utts),
        "hours": round(sum(u.duration for u in utts) / 3600, 2),
        **{k: round(v, 2) for k, v in score(refs, hyps).items()},
        "secs": round(time.time() - t0, 1),
    }
    print(json.dumps(res))


if __name__ == "__main__":
    main()
