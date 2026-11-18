"""Workstream 4: Decoder Lens Spectrum & Probing Integration."""

from ws4_lens.jacobian_lens import (
    estimate_jacobians,
    jacobian_lens,
    load_jacobian_checkpoint,
    save_jacobian_checkpoint,
)
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
from ws4_lens.upos_bridge import build_upos_lexicon, token_to_upos

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
    "estimate_jacobians",
    "jacobian_lens",
    "save_jacobian_checkpoint",
    "load_jacobian_checkpoint",
    "build_upos_lexicon",
    "token_to_upos",
]
