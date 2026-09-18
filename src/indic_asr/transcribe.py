"""Thin wrapper around the NeMo IndicConformer model."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from indic_asr.audio import prepare_wav

MODEL_ID = "ai4bharat/indicconformer_stt_as_hybrid_ctc_rnnt_large"
LANGUAGE_ID = "as"

Decoder = Literal["ctc", "rnnt"]


class Transcriber:
    def __init__(
        self,
        model_id_or_path: str = MODEL_ID,
        device: str | None = None,
    ) -> None:
        # Imported lazily: torch/NeMo are heavy and slow to import.
        import torch
        from dotenv import load_dotenv
        import nemo.collections.asr as nemo_asr

        load_dotenv()  # picks up HF_TOKEN for the gated model repo

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if not model_id_or_path.endswith(".nemo"):
            model_id_or_path = _download_nemo(model_id_or_path)
        self.model = nemo_asr.models.ASRModel.restore_from(model_id_or_path, map_location=self.device)

        self.model.freeze()
        self.model = self.model.to(self.device)

    def transcribe(
        self,
        files: Sequence[str | Path],
        decoder: Decoder = "rnnt",
        batch_size: int = 1,
    ) -> list[str]:
        """Transcribe audio files. Any format/sample rate is accepted; it is
        converted to 16 kHz mono WAV first."""
        wavs = [str(prepare_wav(f)) for f in files]
        self.model.cur_decoder = decoder
        kwargs = {"batch_size": batch_size, "language_id": LANGUAGE_ID}
        if decoder == "ctc":
            kwargs["logprobs"] = False
        import torch

        with torch.inference_mode():
            out = self.model.transcribe(wavs, **kwargs)
        # RNNT decoding returns (best_hyps, all_hyps); CTC returns just the list.
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], list):
            out = out[0]
        if len(out) != len(wavs):
            raise RuntimeError(f"expected {len(wavs)} transcripts, got {len(out)}")
        return [_as_text(o).strip() for o in out]


def _download_nemo(repo_id: str) -> str:
    """Fetch the .nemo checkpoint from an HF repo and return its local path.

    Not using `ASRModel.from_pretrained`: it assumes the file is named
    `<repo_name>.nemo`, which AI4Bharat's repos don't follow.
    """
    from huggingface_hub import HfApi, hf_hub_download

    nemo_files = [f for f in HfApi().list_repo_files(repo_id) if f.endswith(".nemo")]
    if len(nemo_files) != 1:
        raise ValueError(f"expected exactly one .nemo file in {repo_id}, found {nemo_files}")
    return hf_hub_download(repo_id, nemo_files[0])


def _as_text(item) -> str:
    """NeMo returns plain strings in some versions and Hypothesis objects in others."""
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)) and item:
        return _as_text(item[0])
    return getattr(item, "text", str(item))
