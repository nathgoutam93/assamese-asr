"""Pre-download the model weights into the HF cache (optional; from_pretrained does this too)."""

from huggingface_hub import snapshot_download

from assamese_asr.transcribe import MODEL_ID

print(snapshot_download(MODEL_ID))
