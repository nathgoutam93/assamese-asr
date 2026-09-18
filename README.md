# indic-asr

Assamese speech-to-text using
[`ai4bharat/indicconformer_stt_as_hybrid_ctc_rnnt_large`](https://huggingface.co/ai4bharat/indicconformer_stt_as_hybrid_ctc_rnnt_large)
(NeMo Conformer, hybrid CTC + RNNT decoders, ~120M params).

## Setup

Requires Python 3.10–3.12 and system `ffmpeg`/`libsndfile` (`sudo apt install ffmpeg libsndfile1`).

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

The model needs **AI4Bharat's NeMo fork** (`nemo-v2` branch) — upstream NeMo does not
support the `language_id` argument. `pyproject.toml` pulls it from git. If that
install fails (NeMo has a lot of native deps), fall back to the official recipe:

```bash
git clone https://github.com/AI4Bharat/NeMo.git && cd NeMo && git checkout nemo-v2 && bash reinstall.sh
cd .. && uv pip install -e . --no-deps
```

CPU-only machine? Install a CPU torch first to avoid the CUDA wheels:

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

The model repo is **gated**: accept the terms on its Hugging Face page, then put a
read token in `.env` (see `.env.example`):

```
HF_TOKEN=hf_...
```

## Usage

```bash
indic-asr samples/audio.wav                 # RNNT decoder (default, more accurate)
indic-asr samples/audio.wav --decoder ctc   # CTC decoder (faster)
```

```python
from indic_asr import Transcriber

t = Transcriber()                      # downloads weights to the HF cache on first run
print(t.transcribe(["samples/audio.wav"], decoder="rnnt"))
```

Input audio is converted to 16 kHz mono WAV automatically (`indic_asr/audio.py`).

`samples/` holds three Assamese clips from `SPRINGLab/IndicTTS_Assamese` with
reference transcripts in `samples/refs.json`; `python tests/smoke.py` runs both
decoders over them. On CPU, model load takes ~1 min and inference is ~1 s/clip.

## Notes

- `Transcriber` downloads the `.nemo` file itself and calls `restore_from`, because
  `ASRModel.from_pretrained` assumes the file is named `<repo>.nemo` and this repo
  ships `indicconformer_stt_as_hybrid_rnnt_large.nemo`.
- The fork is NeMo 1.23, so `pytorch-lightning<2.5` and `numpy<2` are pinned.

## Layout

```
src/indic_asr/
  audio.py        # resample/downmix to 16k mono
  transcribe.py   # Transcriber: loads model, ctc/rnnt transcription
  cli.py          # `indic-asr` command
scripts/download_model.py
tests/smoke.py    # python tests/smoke.py
```
