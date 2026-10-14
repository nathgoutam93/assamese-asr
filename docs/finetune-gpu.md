# Fine-tuning pruned models on a rented GPU (shrinking plan A, step 3)

Scripts: `scripts/finetune_pruned.py` (train), `scripts/eval_checkpoint.py` (WER),
`scripts/export_onnx.py --checkpoint` (sherpa-onnx export). Background and the
no-training sweep that picked the layer sets: `docs/shrinking-plan.md`,
`experiments/layer_pruning/results.md`.

## Machine

- One NVIDIA GPU, 24 GB+ (A10/L4/3090/4090 fine; A100/H100 faster). bf16 needs
  Ampere or newer; on older cards use `--precision fp16`.
- ~80 GB free disk: 49 GB of IndicVoices-as train parquet + the model + checkpoints.
- 8+ CPU cores: audio is decoded in DataLoader workers (`--workers`).

## Setup

On the GPU box:

```bash
git clone https://github.com/nathgoutam93/assamese-asr.git && cd assamese-asr
echo "HF_TOKEN=hf_..." > .env   # not in the repo; the gated model and dataset need it
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv is missing
sudo apt install -y ffmpeg libsndfile1
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .                                  # PyPI torch wheels include CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

If the NeMo fork fails to install, use the fallback in the README.

Sanity check: this should match the baseline row of
`experiments/layer_pruning/results.md` (same 400 utterances, GPU is just faster):

```bash
python scripts/eval_checkpoint.py
```

## Runs

The first run downloads all 112 train shards (49 GB). To check the pipeline
quickly first, add `--max-shards 2 --max-steps 200 --eval-every 100` (and remove
the run directory before the real run).

Candidates from the sweep (dev WER before any training; baseline 19.84):

```bash
# 1. 13 layers, 89.9M params (43.6 WER untrained)
python scripts/finetune_pruned.py --drop 2,3,8,9 --out runs/drop4-least

# 2. 11 layers, 77.3M params (73.1 WER untrained)
python scripts/finetune_pruned.py --drop 5,6,7,8,9,10 --out runs/drop6-middle

# 3. run 1 with distillation from the unpruned model; compare with run 1
python scripts/finetune_pruned.py --drop 2,3,8,9 --kd-weight 1.0 --out runs/drop4-least-kd
```

Run 1 first. If it gets back to within ~1 WER of the baseline, run 2 shows how far
this goes; if it does not, run 3 shows whether distillation closes the gap.
The training set is roughly 1000 h (the startup line prints the exact figure), so
the default 30k steps at 600 s batches is about 5 epochs.

Useful flags:

| flag | default | notes |
|---|---|---|
| `--batch-seconds` | 600 | padded audio per batch. 24 GB: 400-600; 80 GB: 1200-1600. On OOM halve it and add `--accum 2` to keep the same effective batch |
| `--max-steps` | 30000 | the startup line `~N steps/epoch` helps pick it; 3-5 epochs is a reasonable fine-tune |
| `--lr` | 1e-4 | peak, cosine to 5% after `--warmup` 1000 steps. The original training peaked ~4e-4 |
| `--kd-weight` | 0 | > 0 adds KL to the unpruned model's CTC posteriors (loads a second model, ~30% slower) |
| `--max-duration` | 20 | seconds; longer utterances are skipped |
| `--eval-every` | 1000 | dev WER on the 400-utterance sweep set; best one saved to `best.pt` |

Run under `tmux`/`nohup`. If interrupted, rerun the same command with `--resume`
(continues from `last.pt`; the data stream restarts at a fresh epoch shuffle).

## Watching

`runs/<name>/log.jsonl` gets a line every 50 steps (`ctc` loss, `lr`, `gnorm`,
`audio_h_per_h` = throughput in audio-seconds per wall-second) and every eval
(`dev_wer`, `best_wer`). Step 0 logs the pruned model's WER before training, which
should match the sweep table.

```bash
tail -f runs/drop4-least/log.jsonl | grep dev_wer
```

## After training

```bash
# final numbers on the whole valid split (~5k utterances), baseline first
python scripts/eval_checkpoint.py --n 0
python scripts/eval_checkpoint.py --n 0 --checkpoint runs/drop4-least/best.pt

# sherpa-onnx export; same tokens.txt and Android config as the full model
python scripts/export_onnx.py --checkpoint runs/drop4-least/best.pt --out export/drop4-least
python scripts/test_onnx.py --model export/drop4-least/model.int8.onnx --tokens export/drop4-least/tokens.txt
```

Bring back `best.pt` (~360 MB for 13 layers) or just the `export/<name>/`
directory. A 13-layer int8 export is ~120 MB (full model: 145 MB).

The 400-utterance dev set also picks `best.pt`, so it is slightly optimistic on
it; the `--n 0` numbers on the whole valid split are the ones to compare.
