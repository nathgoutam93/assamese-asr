"""Assamese-only CTC view of the hybrid model, shared by export and pruning."""

from __future__ import annotations

import torch

from assamese_asr.transcribe import LANGUAGE_ID


def build_assamese_ctc_head(model):
    """Return a ConvASRDecoder holding only the Assamese rows (+ blank) of the CTC head."""
    from nemo.collections.asr.modules import ConvASRDecoder

    mask = torch.tensor(model.language_masks[LANGUAGE_ID], dtype=torch.bool)
    idx = mask.nonzero().squeeze(1)
    n_tok = len(idx) - 1  # last selected row is the shared blank
    # The masked rows must be a contiguous block followed by blank, otherwise the
    # mapping from row -> token id would not be the tokenizer's id order.
    assert torch.equal(idx[:-1], torch.arange(idx[0], idx[0] + n_tok)), "non-contiguous language block"
    assert idx[-1] == mask.numel() - 1, "blank is expected to be the last row"

    sp = model.tokenizer.tokenizers_dict[LANGUAGE_ID]
    vocab = [sp.ids_to_tokens([i])[0] for i in range(sp.vocab_size)]
    assert len(vocab) == n_tok, (len(vocab), n_tok)

    old = model.ctc_decoder.decoder_layers[0]
    head = ConvASRDecoder(feat_in=old.in_channels, num_classes=n_tok, vocabulary=vocab)
    new = head.decoder_layers[0]
    with torch.no_grad():
        new.weight.copy_(old.weight[idx])
        new.bias.copy_(old.bias[idx])
    return head, vocab
