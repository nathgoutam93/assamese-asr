"""Export an Assamese-only CTC ONNX model for on-device use (sherpa-onnx).

What this does, and why:
- The checkpoint is a hybrid RNNT+CTC model with a 22-language aggregate
  vocabulary (22 x 256 tokens + blank). Only the CTC branch and the Assamese
  slice of the output layer are kept: encoder unchanged, CTC head cut from
  Conv1d(512 -> 5633) to Conv1d(512 -> 257). Same logits, no retraining.
- The RNNT prediction network and joint are dropped (~11M params) and greedy
  CTC decoding is much cheaper at runtime than transducer search.
- Output layout follows sherpa-onnx's NeMo CTC recipe: fp32 model.onnx,
  dynamically-quantized int8 model.int8.onnx, and tokens.txt with <blk> last.
  Check the result with scripts/test_onnx.py.

With --checkpoint, exports a layer-pruned model from scripts/finetune_pruned.py
instead of the full encoder.

Usage: python scripts/export_onnx.py [--out export/] [--checkpoint runs/x/best.pt]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from indic_asr.pruning import load_pruned_model
from indic_asr.transcribe import LANGUAGE_ID


def _use_legacy_onnx_exporter() -> None:
    """torch>=2.9 defaults to the dynamo exporter; NeMo 1.23 targets the TorchScript one."""
    import functools

    orig = torch.onnx.export
    torch.onnx.export = functools.partial(orig, dynamo=False)


def quantize_int8(src: Path, dst: Path) -> None:
    """Dynamic int8 for MatMul and pointwise Conv only.

    Measured on samples/: quantizing the depthwise convs (Conformer conv module)
    and the subsampling convs changes transcripts and is ~5x slower in ORT
    (ConvInteger). With them excluded, output matches fp32 exactly.
    """
    m = onnx.load(str(src))
    exclude = [
        n.name
        for n in m.graph.node
        if n.op_type == "Conv" and ("depthwise" in n.name or "pre_encode" in n.name)
    ]
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Conv"],
        nodes_to_exclude=exclude,
    )


def add_metadata(path: Path, meta: dict[str, str]) -> None:
    m = onnx.load(str(path))
    del m.metadata_props[:]
    for k, v in meta.items():
        m.metadata_props.add(key=k, value=str(v))
    onnx.save(m, str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="export", type=Path)
    ap.add_argument("--checkpoint", help="pruned checkpoint from scripts/finetune_pruned.py")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = load_pruned_model(args.checkpoint, "cpu")  # encoder + Assamese CTC head, eval mode
    head = model.ctc_decoder
    vocab = list(head.vocabulary)

    fp32 = args.out / "model.onnx"
    _use_legacy_onnx_exporter()
    model.export(str(fp32), check_trace=False)

    enc_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = sum(p.numel() for p in head.parameters())
    print(f"exported params: encoder {enc_params/1e6:.1f}M + ctc head {head_params/1e6:.2f}M")

    meta = {
        "vocab_size": len(vocab) + 1,
        "normalize_type": model.cfg.preprocessor.normalize,
        "subsampling_factor": model.cfg.encoder.subsampling_factor,
        "model_type": "EncDecCTCModelBPE",
        "version": "1",
        "model_author": "AI4Bharat (exported via indic-asr)",
        "url": "https://huggingface.co/ai4bharat/indicconformer_stt_as_hybrid_ctc_rnnt_large",
        "language": LANGUAGE_ID,
        "comment": "Assamese-only CTC branch of IndicConformer hybrid CTC/RNNT large, "
        f"{len(model.encoder.layers)} of 17 encoder layers",
    }
    add_metadata(fp32, meta)

    int8 = args.out / "model.int8.onnx"
    quantize_int8(fp32, int8)
    add_metadata(int8, meta)

    with open(args.out / "tokens.txt", "w", encoding="utf-8") as f:
        for i, tok in enumerate(vocab):
            f.write(f"{tok} {i}\n")
        f.write(f"<blk> {len(vocab)}\n")

    for p in (fp32, int8):
        print(f"{p}: {p.stat().st_size/1e6:.1f} MB")
    print(f"tokens: {len(vocab) + 1}")


if __name__ == "__main__":
    main()
