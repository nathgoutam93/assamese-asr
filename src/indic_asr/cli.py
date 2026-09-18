import argparse
import sys

MODEL_ID = "ai4bharat/indicconformer_stt_as_hybrid_ctc_rnnt_large"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="indic-asr", description="Assamese speech-to-text")
    p.add_argument("audio", nargs="+", help="audio file(s); any format, auto-converted to 16k mono")
    p.add_argument("--decoder", choices=["ctc", "rnnt"], default="rnnt")
    p.add_argument("--model", default=MODEL_ID, help="HF model id or local .nemo path")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args(argv)

    from indic_asr.transcribe import Transcriber

    t = Transcriber(args.model, device=args.device)
    for path, text in zip(args.audio, t.transcribe(args.audio, args.decoder, args.batch_size)):
        print(f"{path}\t{text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
