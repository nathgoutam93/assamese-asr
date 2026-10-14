# Android app handoff

Paste the "Brief" section into a Claude Code session started in a NEW, EMPTY
directory (e.g. `mkdir ~/assamese-asr-android && cd ~/assamese-asr-android && claude`).
Model files come from this repo's `export/` directory (regenerate with
`python scripts/export_onnx.py` if missing).

---

## Brief

Scaffold an Android app (Kotlin, Gradle Kotlin DSL, min SDK 24, target latest
stable) that does offline Assamese speech-to-text with sherpa-onnx. No network
calls; everything runs on device.

### Model (already exported, do not re-export)

Copy these three files from `/home/gnrc/indic-conformer-as-120m/export/`:

| file | size | what |
|---|---|---|
| `model.int8.onnx` | 145 MB | NeMo Conformer-CTC, Assamese only, int8 |
| `tokens.txt` | 257 lines | SentencePiece pieces, format `<piece> <id>`, `<blk>` is last (id 256) |
| `model.onnx` | 482 MB | fp32 reference; do NOT ship, only for A/B testing |

Model facts sherpa-onnx needs:
- type: NeMo EncDecCTC (`modelType = "nemo_ctc"`), offline/non-streaming
- input audio: 16 kHz mono float PCM; sherpa-onnx does the 80-dim fbank + per-feature normalization itself (metadata `normalize_type=per_feature`, `subsampling_factor=4` is embedded in the ONNX)
- decoding: greedy search is sufficient (output already validated to match the original PyTorch model exactly)
- CPU inference; use 2-4 threads. No GPU/NNAPI delegate needed.

The 145 MB file is too big to bundle comfortably in the APK. Ship it either as an
Android App Bundle asset pack (install-time) or download on first launch from a
URL the user provides later; make this a pluggable `ModelProvider` and start with
the simplest thing that works locally (put it under `app/src/main/assets/` and
copy to `filesDir` on first run, since sherpa-onnx needs a real file path).

### Runtime: sherpa-onnx

- Get the prebuilt `sherpa-onnx-v<version>.aar` from
  https://github.com/k2-fsa/sherpa-onnx/releases (use the latest; the Python
  package that validated the export is 1.13.8) and put it in `app/libs/`, or
  use the Maven artifact if one is available for that version. Check the
  sherpa-onnx Android docs for the current recommended way:
  https://k2-fsa.github.io/sherpa/onnx/android/index.html
- The Kotlin API lives in the sherpa-onnx repo under
  `android/SherpaOnnx/app/src/main/java/com/k2fsa/sherpa/onnx/` (`OfflineRecognizer.kt`,
  `FeatureConfig.kt`, ...). If the AAR does not include the classes, copy those files in.
- Config to use (Kotlin):

```kotlin
val config = OfflineRecognizerConfig(
    featConfig = FeatureConfig(sampleRate = 16000, featureDim = 80),
    modelConfig = OfflineModelConfig(
        nemo = OfflineNemoEncDecCtcModelConfig(model = "<abs path>/model.int8.onnx"),
        tokens = "<abs path>/tokens.txt",
        numThreads = 2,
        debug = false,
        modelType = "nemo_ctc",
    ),
    decodingMethod = "greedy_search",
)
val recognizer = OfflineRecognizer(config = config)   // or assetManager overload
// per utterance:
val stream = recognizer.createStream()
stream.acceptWaveform(samples /* FloatArray, -1..1 */, sampleRate = 16000)
recognizer.decode(stream)
val text = recognizer.getResult(stream).text
stream.release()
```

### App scope (v1)

1. Single screen: big record button (press to start, press to stop), transcript
   text view, model load status.
2. Record with `AudioRecord` at 16 kHz mono PCM16, convert to float, hand the
   whole utterance to the recognizer on stop (the model is offline, not streaming;
   do not try to get partial results).
3. Model loading on a background thread with a progress state; the recognizer is a
   singleton kept for the app lifetime (load takes ~1-2 s, ~300 MB RAM).
4. Show decode time and audio duration for each utterance (RTF) in the UI; this is
   the number we care about for evaluating the model on real hardware.
5. `RECORD_AUDIO` runtime permission handling.
6. A debug menu entry to transcribe a bundled WAV so behaviour can be checked
   without a microphone. Bundle `samples/as_0.wav` from the model repo
   (`/home/gnrc/indic-conformer-as-120m/samples/`, 48 kHz - resample to 16 kHz
   on load or pre-convert with `ffmpeg -i as_0.wav -ar 16000 -ac 1 as_0_16k.wav`).
   Expected transcript: `এদিন মাছবোৰ গোট খাই মাছখোৱা জন্তুবোৰৰ সৈতে`

Keep it plain: single Activity + ViewModel, Jetpack Compose or XML (either is fine),
no DI framework. Assamese text renders fine with the system font; make sure the
transcript TextView is selectable.

### Done when

- `./gradlew assembleDebug` succeeds
- On a device, the bundled sample transcribes to the expected text and the
  decode time is displayed
- Mic recording produces a transcript

---

## Notes for this repo

- Model regeneration: `python scripts/export_onnx.py`; validation: `python scripts/test_onnx.py`.
- If the export changes (e.g. after pruning per `docs/shrinking-plan.md`), only the
  three files change; the Android config stays the same as long as it is still a
  NeMo CTC model with 4x subsampling.
