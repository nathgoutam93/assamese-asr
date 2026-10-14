"""Assamese ASR using AI4Bharat IndicConformer (hybrid CTC/RNNT)."""

MODEL_ID = "ai4bharat/indicconformer_stt_as_hybrid_ctc_rnnt_large"


def __getattr__(name):
    if name == "Transcriber":  # lazy: avoids importing torch/NeMo on package import
        from assamese_asr.transcribe import Transcriber

        return Transcriber
    raise AttributeError(name)


__all__ = ["MODEL_ID", "Transcriber"]
