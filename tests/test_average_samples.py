"""Tests for the ``average_samples`` flag of ``auto_regressive_inference``.

Uses minimal fake tokenizer / model and a stubbed ``sample_from_logits`` so the
generation loop runs without downloading any pretrained weights. Verifies that:
  - ``average_samples=False`` keeps the per-sample dimension;
  - ``average_samples=True`` (default) collapses it (backward-compatible shape);
  - the averaged output equals the mean over samples of the per-sample output.
"""
from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import model.kronos as kronos_mod  # noqa: E402
from model.kronos import auto_regressive_inference  # noqa: E402

_BATCH = 1
_SEQ = 4
_PRED = 2
_SAMPLES = 3
_FEAT = 6
_STAMP = 4
_MAX_CTX = 8       # >= seq + pred so the buffer never rolls (simple path)
_VOCAB = 8


@dataclass
class _FakeTokenizer:
    def encode(self, x, half=False):
        bsz, slen = x.size(0), x.size(1)
        return [
            torch.zeros(bsz, slen, dtype=torch.long),
            torch.zeros(bsz, slen, dtype=torch.long),
        ]

    def decode(self, input_tokens, half=False):
        pre = input_tokens[0]
        return torch.zeros(pre.size(0), pre.size(1), _FEAT)


@dataclass
class _FakeModel:
    def decode_s1(self, pre, post, stamp):
        bsz, slen = pre.size(0), pre.size(1)
        return torch.zeros(bsz, slen, _VOCAB), torch.zeros(bsz, slen, 4)

    def decode_s2(self, context, sample_pre):
        return torch.zeros(context.size(0), context.size(1), _VOCAB)


def _fake_sample_from_logits(logits, temperature=1.0, top_k=0, top_p=1.0, sample_logits=True):
    return torch.zeros(logits.size(0), 1, dtype=torch.long)


def _inputs():
    return (
        torch.zeros(_BATCH, _SEQ, _FEAT),
        torch.zeros(_BATCH, _SEQ, _STAMP),
        torch.zeros(_BATCH, _PRED, _STAMP),
    )


@pytest.fixture
def patched_sampler(monkeypatch):
    monkeypatch.setattr(kronos_mod, "sample_from_logits", _fake_sample_from_logits)


def _run(average_samples):
    x, x_stamp, y_stamp = _inputs()
    return auto_regressive_inference(
        _FakeTokenizer(), _FakeModel(), x, x_stamp, y_stamp,
        max_context=_MAX_CTX, pred_len=_PRED, sample_count=_SAMPLES,
        average_samples=average_samples,
    )


def test_average_samples_false_keeps_sample_dimension(patched_sampler):
    preds = _run(average_samples=False)
    assert preds.shape == (_BATCH, _SAMPLES, _SEQ + _PRED, _FEAT)


def test_average_samples_true_collapses_sample_dimension(patched_sampler):
    preds = _run(average_samples=True)
    assert preds.shape == (_BATCH, _SEQ + _PRED, _FEAT)


def test_averaged_output_is_mean_over_samples(patched_sampler):
    paths = _run(average_samples=False)
    averaged = _run(average_samples=True)
    np.testing.assert_allclose(paths.mean(axis=1), averaged, rtol=1e-6, atol=1e-6)
