"""Run the exported model through sherpa-onnx on samples/ and compare to refs.

Usage: python scripts/test_onnx.py [--model export/model.int8.onnx]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sherpa_onnx

from indic_asr.audio import prepare_wav


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="export/model.int8.onnx")
    ap.add_argument("--tokens", default="export/tokens.txt")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=args.model, tokens=args.tokens, num_threads=args.threads, decoding_method="greedy_search"
    )
    print(f"loaded {args.model} in {time.time() - t0:.1f}s")

    import soundfile as sf

    refs = json.loads(Path("samples/refs.json").read_text())
    for f, ref in refs.items():
        audio, sr = sf.read(str(prepare_wav(f)), dtype="float32")
        s = rec.create_stream()
        s.accept_waveform(sr, audio)
        t0 = time.time()
        rec.decode_stream(s)
        dt = time.time() - t0
        hyp = s.result.text.strip()
        mark = "OK " if hyp == ref else "~~ "
        print(f"{mark}{f}  ({len(audio)/sr:.1f}s audio, {dt:.2f}s decode)\n   hyp: {hyp}\n   ref: {ref}")


if __name__ == "__main__":
    main()
