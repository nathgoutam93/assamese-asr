# assamese-asr

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
assamese-asr samples/audio.wav                 # RNNT decoder (default, more accurate)
assamese-asr samples/audio.wav --decoder ctc   # CTC decoder (faster)
```

```python
from assamese_asr import Transcriber

t = Transcriber()                      # downloads weights to the HF cache on first run
print(t.transcribe(["samples/audio.wav"], decoder="rnnt"))
```

Input audio is converted to 16 kHz mono WAV automatically (`assamese_asr/audio.py`).

`samples/` holds three Assamese clips from `SPRINGLab/IndicTTS_Assamese` with
reference transcripts in `samples/refs.json`; `python tests/smoke.py` runs both
decoders over them. On CPU, model load takes ~1 min and inference is ~1 s/clip.

## On-device export (Android via sherpa-onnx)

```bash
uv pip install -e ".[export]"
python scripts/export_onnx.py          # -> export/model.onnx, model.int8.onnx, tokens.txt
python scripts/test_onnx.py            # runs export/model.int8.onnx through sherpa-onnx on samples/
```

What the export keeps: the Conformer encoder (115M params) and the **Assamese
slice** of the CTC head (256 tokens + blank). The RNNT branch and the other 21
languages' output rows are dropped. Logits are identical to the original model.

| file | size | notes |
|---|---|---|
| `export/model.onnx` | 482 MB | fp32, output matches NeMo CTC exactly |
| `export/model.int8.onnx` | 145 MB | int8 MatMul + pointwise Conv; output matches fp32 exactly on samples |
| `export/tokens.txt` | — | 257 lines, `<blk>` last |

Quantizing the depthwise/subsampling convs too gets to 138 MB but changes
transcripts and runs ~5x slower in onnxruntime, so those stay fp32.

On Android: use the `sherpa-onnx` AAR and `OfflineRecognizer` with
`OfflineNemoEncDecCtcModelConfig(model = "model.int8.onnx")` plus `tokens.txt`.
Feature extraction (80-mel, per-feature normalization) is handled by sherpa-onnx.

## Notes

- `Transcriber` downloads the `.nemo` file itself and calls `restore_from`, because
  `ASRModel.from_pretrained` assumes the file is named `<repo>.nemo` and this repo
  ships `indicconformer_stt_as_hybrid_rnnt_large.nemo`.
- The fork is NeMo 1.23, so `pytorch-lightning<2.5` and `numpy<2` are pinned.

## Layout

```
src/assamese_asr/
  audio.py        # resample/downmix to 16k mono
  transcribe.py   # Transcriber: loads model, ctc/rnnt transcription
  cli.py          # `assamese-asr` command
  ctc.py          # Assamese slice of the CTC head
  pruning.py      # layer removal, IndicVoices loading, WER scoring, checkpoints
scripts/download_model.py
scripts/export_onnx.py   # Assamese-only CTC ONNX export + int8 (--checkpoint for pruned models)
scripts/test_onnx.py     # sherpa-onnx check of the export
scripts/prune_layers.py      # no-training layer-pruning sweep
scripts/finetune_pruned.py   # fine-tune a pruned model (GPU; docs/finetune-gpu.md)
scripts/eval_checkpoint.py   # WER of base model or pruned checkpoint
tests/smoke.py    # python tests/smoke.py
```

Model shrinking beyond the export: `docs/shrinking-plan.md`.
