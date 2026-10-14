# Shrinking the model further (option 2: reduce parameters)

Status: **A in progress** (steps 1-2 done, see below). Option 1 (CTC-only, Assamese-only head, int8) is done in
`scripts/export_onnx.py` and lands at 145 MB / 115M params. Everything below needs
training and an Assamese dataset; it is recorded here for later experiments.

## Where the parameters are

| part | params | notes |
|---|---|---|
| Conformer encoder | 115.1M | 17 layers, d_model 512, 8 heads, ff x4, conv kernel 31, 4x striding subsampling |
| RNNT prediction net | 6.9M | already dropped in export |
| RNNT joint | 4.4M | already dropped in export |
| CTC head | 2.9M -> 0.13M | already sliced to Assamese (256 + blank) |

The encoder is dense and uniformly loaded: there is no free slack to cut without
retraining. Per-layer FLOPs are roughly equal, so size and latency scale ~linearly
with layer count.

Config source: `model_config.yaml` inside the `.nemo` archive
(`tar xOf <file>.nemo ./model_config.yaml`).

## Data

- **IndicVoices** (AI4Bharat, https://huggingface.co/datasets/ai4bharat/IndicVoices)
  is what the model was trained on; the Assamese subset is the natural fine-tuning
  and eval set. Gated, same login as the model.
- `SPRINGLab/IndicTTS_Assamese` (used for `samples/`) is clean read TTS speech: fine
  for sanity checks, too easy for measuring WER.
- Always hold out a test split and report WER of the **unmodified** model on it
  first, so every experiment has a baseline.

## Experiments, cheapest first

### A. Layer pruning + short fine-tune  (expected: 115M -> 70-85M)

1. Drop the top N encoder layers (`model.encoder.layers = layers[:17-N]`), N in {4, 6, 8}.
2. Re-attach the sliced CTC head, measure WER with **no** training: shows how much
   each cut costs before any recovery.
3. Fine-tune encoder + CTC head only (RNNT branch removed; plain CTC loss) on
   IndicVoices-as for a few epochs, small LR (1e-5 to 5e-5), SpecAugment on.
4. Expect: N=4-6 recoverable to within ~1 WER; N>=8 loses noticeably.

Also worth trying: drop *middle* layers instead of top ones and compare, and
measure per-layer importance by zeroing one layer at a time (no training needed).

**Steps 1-2 results** (`python scripts/prune_layers.py`, full table in
`experiments/layer_pruning/results.md`). Eval: 400 random IndicVoices-as `valid`
utterances (0.69 h, seed 0; `<tag>`-only references excluded); baseline
19.84 WER / 6.80 CER.

- Dropping top layers is the worst choice: the CTC head only reads the last
  layer's representation (top-16 already +21 WER, top-13 ~98 WER).
- Single-layer importance: layers 16, 15, 12 (+16-21 WER), 6, 0, 11 (+7-9) matter;
  1-5 and 7-9 cost only +1.4-3.9 WER each.
- Costs do not add up linearly: best 4-layer drop is +24 WER (`least` = {2,3,8,9}),
  and at 6 layers contiguous middle {5..10} (73 WER) beats the one-at-a-time
  ranking (94 WER). No cut is usable without step 3.
- Candidates for fine-tuning: drop {2,3,8,9} (13 layers, 89.9M) and drop {5..10}
  (11 layers, 77.3M). Removing from layers 1-9 and keeping 0 and 10-16 is the
  pattern to follow; iterative greedy removal would rank better than one-shot.

Step 3: `scripts/finetune_pruned.py`, run on a rented GPU per `docs/finetune-gpu.md`.

### B. Knowledge distillation into a small student  (expected: 15-30M)

Student candidates (all exportable to sherpa-onnx):
- Conformer 8-12 layers, d_model 256, ff x4, kernel 31 (~13-20M)
- Squeezeformer or Zipformer-S (~20-30M) if using icefall instead of NeMo

Distillation targets, combine:
- CTC loss on ground truth
- KL between student and teacher CTC posteriors (frame-level; both use 4x subsampling
  so frames align)
- optionally L2 on encoder outputs through a linear projection (512 -> 256)

Initialise the student encoder from the teacher where shapes allow (subsampling,
attention of a subset of layers projected down) to speed convergence.

This is a real training run: budget days on a single GPU with IndicVoices-as
(~hundreds of hours). Realistic floor without serious WER loss is 20-30M params.

### C. Structured pruning (head / FF pruning)

Remove attention heads and FF neurons by importance (activation magnitude or
gradient-based), then fine-tune. Typically 20-40% reduction at <1 WER, less
dramatic than A or B but composable with A.

### D. Quantization beyond int8

- int4 weight-only for MatMul: halves the size again (~75 MB from the 145 MB
  model) but onnxruntime mobile support for MatMulNBits on ARM is uneven; test on
  the actual device.
- QAT (quantization-aware training) only after A/B, to recover the last bit.
- Keep the finding from option 1: never quantize the depthwise or subsampling
  convs (accuracy loss + 5x slower in ORT).

### E. Streaming

Not a size change, but relevant for phones: this checkpoint is offline
(full-context attention). Streaming needs a cache-aware Conformer with limited
context; that requires retraining/fine-tuning with chunked attention masks. If
you do B, train the student as streaming from the start.

## Evaluation checklist for every experiment

- WER on the held-out IndicVoices-as test split (teacher baseline first)
- exported int8 ONNX size
- RTF on the target phone via sherpa-onnx (not on the dev machine)
- peak RSS on device
