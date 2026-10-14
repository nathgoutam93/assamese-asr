"""Encoder layer pruning: build Assamese CTC models with fewer Conformer layers
and score them on IndicVoices. See docs/shrinking-plan.md (experiment A)."""

from __future__ import annotations

import contextlib
import io
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import torch

from indic_asr.ctc import build_assamese_ctc_head
from indic_asr.transcribe import LANGUAGE_ID, Transcriber

IV_REPO = "ai4bharat/IndicVoices"
IV_LANG_DIR = "assamese"


@dataclass
class Utterance:
    id: str
    audio: np.ndarray  # float32, 16 kHz mono
    text: str
    duration: float


# --------------------------------------------------------------------------- model


def load_ctc_model(device: str = "cpu"):
    """The hybrid checkpoint reduced to encoder + Assamese CTC head, in eval mode.

    Same setup as scripts/export_onnx.py, so WER measured here is what the
    exported ONNX model would get (before int8).
    """
    model = Transcriber(device=device).model
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    head, _ = build_assamese_ctc_head(model)
    model.ctc_decoder = head.to(device)
    model.cur_decoder = "ctc"
    assert model.encoder.out_proj is None, "layer-output scoring assumes no encoder out_proj"
    model.eval()
    return model


def keep_layers(model, keep: Sequence[int]) -> None:
    """Keep only encoder layers `keep` (original indices, in order), in place.

    The encoder zips `layer_drop_probs` with `layers`, so both are cut together;
    cfg is updated so a saved .nemo restores with the new depth.
    """
    enc = model.encoder
    keep = list(keep)
    enc.layers = torch.nn.ModuleList(enc.layers[i] for i in keep)
    enc.layer_drop_probs = [enc.layer_drop_probs[i] for i in keep]
    enc.n_layers = len(keep)
    with _cfg_writable(model.cfg):
        model.cfg.encoder.n_layers = len(keep)


@contextlib.contextmanager
def layers_kept(model, keep: Sequence[int]) -> Iterator[None]:
    """Temporarily run with only `keep` layers; restores the full encoder on exit."""
    enc = model.encoder
    saved = (enc.layers, enc.layer_drop_probs, enc.n_layers, model.cfg.encoder.n_layers)
    keep_layers(model, keep)
    try:
        yield
    finally:
        enc.layers, enc.layer_drop_probs, enc.n_layers, n = saved
        with _cfg_writable(model.cfg):
            model.cfg.encoder.n_layers = n


@contextlib.contextmanager
def _cfg_writable(cfg):
    from omegaconf import open_dict

    with open_dict(cfg):
        yield


def save_checkpoint(path, model, keep: Sequence[int], **extra) -> None:
    """Pruned encoder + Assamese CTC head. Restore with `load_pruned_model`.

    Not a .nemo: the hybrid model's config describes the 22-language head, which
    no longer matches. The base checkpoint supplies preprocessor/tokenizer/config.
    """
    torch.save(
        {
            "keep": list(keep),
            "encoder": model.encoder.state_dict(),
            "ctc_decoder": model.ctc_decoder.state_dict(),
            **extra,
        },
        path,
    )


def load_pruned_model(checkpoint=None, device: str = "cpu"):
    """`load_ctc_model`, then cut to the checkpoint's layers and load its weights."""
    model = load_ctc_model(device)
    if checkpoint:
        ck = torch.load(checkpoint, map_location=device, weights_only=True)
        keep_layers(model, ck["keep"])
        model.encoder.load_state_dict(ck["encoder"])
        model.ctc_decoder.load_state_dict(ck["ctc_decoder"])
    return model


def count_params(model) -> int:
    """Params that ship in the export: encoder + Assamese CTC head."""
    return sum(p.numel() for m in (model.encoder, model.ctc_decoder) for p in m.parameters())


# --------------------------------------------------------------------------- data


def iv_shards(split: str = "train", max_shards: int | None = None) -> list[str]:
    """Local paths of the IndicVoices-as parquet shards for `split`, downloading as needed."""
    from dotenv import load_dotenv
    from huggingface_hub import HfApi, hf_hub_download

    load_dotenv(".env")  # HF_TOKEN; dataset is gated
    files = sorted(
        f
        for f in HfApi().list_repo_files(IV_REPO, repo_type="dataset")
        if f.startswith(f"{IV_LANG_DIR}/{split}-") and f.endswith(".parquet")
    )
    if not files:
        raise ValueError(f"no {split} shards for {IV_LANG_DIR} in {IV_REPO}")
    if max_shards:
        files = files[:max_shards]
    return [hf_hub_download(IV_REPO, f, repo_type="dataset") for f in files]


def decode_audio(audio_bytes: bytes) -> np.ndarray:
    """Parquet audio bytes -> float32 16 kHz mono."""
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16_000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
    return audio


def load_indicvoices(
    split: str = "valid", n: int | None = None, seed: int = 0, max_duration: float = 30.0
) -> list[Utterance]:
    """IndicVoices Assamese utterances from the HF parquet shards.

    `n` draws a fixed random subset (same `seed` -> same subset), so numbers from
    different runs are comparable.
    """
    import pyarrow.parquet as pq

    rows = []
    for path in iv_shards(split):
        name = "/".join(path.split("/")[-2:])
        t = pq.read_table(path, columns=["audio_filepath", "text", "duration"])
        for i, r in enumerate(t.to_pylist()):
            if r["duration"] <= max_duration and normalize(r["text"]):
                rows.append((f"{name}:{i}", r))
    if n:
        rows = random.Random(seed).sample(rows, min(n, len(rows)))

    out = []
    for uid, r in rows:
        audio = decode_audio(r["audio_filepath"]["bytes"])
        out.append(Utterance(uid, audio, r["text"], len(audio) / 16_000))
    return out


def batches(utts: Sequence[Utterance], max_seconds: float = 120.0) -> Iterator[list[Utterance]]:
    """Duration-sorted batches of at most `max_seconds` of padded audio."""
    utts = sorted(utts, key=lambda u: u.duration)
    batch: list[Utterance] = []
    for u in utts:
        if batch and u.duration * (len(batch) + 1) > max_seconds:
            yield batch
            batch = []
        batch.append(u)
    if batch:
        yield batch


# --------------------------------------------------------------------------- decode / score


@torch.inference_mode()
def encode_batch(model, batch: Sequence[Utterance], capture_layers: bool = False):
    """Run preprocessor + encoder. Returns (encoded [B,D,T], lengths, per-layer outputs).

    With `capture_layers`, the third value holds every layer's output as [B,D,T],
    i.e. what the encoder would return if it stopped after that layer.
    """
    dev = next(model.parameters()).device
    lens = torch.tensor([len(u.audio) for u in batch], device=dev)
    x = torch.zeros(len(batch), int(lens.max()), device=dev)
    for i, u in enumerate(batch):
        x[i, : len(u.audio)] = torch.from_numpy(u.audio)
    feats, flen = model.preprocessor(input_signal=x, length=lens)

    captured: list[torch.Tensor] = []
    hooks = []
    if capture_layers:
        for layer in model.encoder.layers:
            hooks.append(layer.register_forward_hook(lambda m, a, o: captured.append(o.transpose(1, 2))))
    try:
        enc, elen = model.encoder(audio_signal=feats, length=flen)
    finally:
        for h in hooks:
            h.remove()
    return enc, elen, captured


@torch.inference_mode()
def ctc_greedy(model, enc: torch.Tensor, elen: torch.Tensor) -> list[str]:
    logp = model.ctc_decoder(encoder_output=enc)  # [B,T,V], blank = V-1
    blank = logp.shape[-1] - 1
    sp = model.tokenizer.tokenizers_dict[LANGUAGE_ID]
    out = []
    for ids, n in zip(logp.argmax(-1).cpu().tolist(), elen.cpu().tolist()):
        toks, prev = [], None
        for t in ids[:n]:
            if t != prev and t != blank:
                toks.append(t)
            prev = t
        out.append(sp.ids_to_text(toks).strip() if toks else "")
    return out


def normalize(text: str) -> str:
    """NFC, drop <tags> (e.g. <unintelligible>) and punctuation, collapse whitespace.
    Applied to refs and hyps alike. Apostrophes become spaces like other
    punctuation: the model writes হ'ব as `হ ব` (checked on IndicVoices valid)."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.lower().split())


def target_chars(model) -> set[str]:
    """Characters the Assamese tokenizer can produce (excluding <unk>)."""
    sp = model.tokenizer.tokenizers_dict[LANGUAGE_ID]
    toks = [sp.ids_to_tokens([i])[0] for i in range(sp.vocab_size)]
    return {c for t in toks if t != "<unk>" for c in t.replace("\u2581", "")}


def clean_target(text: str, allowed: set[str]) -> str | None:
    """`normalize`d transcript, or None if it is empty or needs characters outside
    the vocabulary (e.g. Latin words): those would train the model to emit <unk>."""
    t = normalize(text)
    return t if t and set(t.replace(" ", "")) <= allowed else None


def score(refs: Sequence[str], hyps: Sequence[str]) -> dict[str, float]:
    import jiwer

    r = [normalize(t) for t in refs]
    h = [normalize(t) for t in hyps]
    return {"wer": 100 * jiwer.wer(r, h), "cer": 100 * jiwer.cer(r, h)}
