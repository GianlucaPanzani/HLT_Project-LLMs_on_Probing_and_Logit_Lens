"""Workstream 4: Decoder Lens Spectrum & Probing Integration."""

from ws4_lens.logit_lens import logit_lens, run_model
from ws4_lens.metrics import entropy, kl_to_final, top1_accuracy
from ws4_lens.tuned_lens import (
    TunedLensTranslators,
    evaluate_dev_kl,
    load_official_tuned_lens,
    load_tuned_lens_checkpoint,
    save_tuned_lens_checkpoint,
    train_tuned_lens,
    tuned_lens,
)

__all__ = [
    "run_model",
    "logit_lens",
    "entropy",
    "kl_to_final",
    "top1_accuracy",
    "TunedLensTranslators",
    "load_official_tuned_lens",
    "tuned_lens",
    "evaluate_dev_kl",
    "train_tuned_lens",
    "save_tuned_lens_checkpoint",
    "load_tuned_lens_checkpoint",
]
