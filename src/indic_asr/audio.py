"""Audio preprocessing: the model expects 16 kHz mono WAV."""

from __future__ import annotations

import tempfile
from pathlib import Path

import librosa
import soundfile as sf

TARGET_SR = 16_000


def prepare_wav(path: str | Path, out_dir: str | Path | None = None) -> Path:
    """Return a path to a 16 kHz mono WAV for `path`.

    If the input already satisfies the requirements it is returned unchanged;
    otherwise it is resampled/downmixed and written to `out_dir` (a temp dir
    by default).
    """
    path = Path(path)
    info = sf.info(str(path))
    if path.suffix.lower() == ".wav" and info.samplerate == TARGET_SR and info.channels == 1:
        return path

    audio, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="indic_asr_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}_16k.wav"
    sf.write(str(out), audio, TARGET_SR, subtype="PCM_16")
    return out
