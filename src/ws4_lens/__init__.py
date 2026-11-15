"""Workstream 4: Decoder Lens Spectrum & Probing Integration."""

from ws4_lens.logit_lens import logit_lens, run_model
from ws4_lens.metrics import entropy, kl_to_final, top1_accuracy

__all__ = [
    "run_model",
    "logit_lens",
    "entropy",
    "kl_to_final",
    "top1_accuracy",
]
