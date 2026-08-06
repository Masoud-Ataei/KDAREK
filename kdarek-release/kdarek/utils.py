"""Small reusable helpers shared across the notebooks.

None of these existed as standalone functions anywhere in the source
repo -- `count_parameters` replaces the inline
`sum(p.numel() for p in model.parameters())` pattern used ad hoc in
Exp0_Cos_SameParms/print_table.py; `set_seed` and `check_error_violation`
formalize the pattern used inline in the Fig2_cos_k2dk_journal*.ipynb
results-table cells.
"""
import random

import numpy as np
import torch


def count_parameters(model, trainable_only=False):
    """Total (or trainable-only) parameter count of a torch module."""
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def set_seed(seed):
    """Reseed torch/numpy/random together -- call right before constructing
    and fitting a model so stochastic models are reproducible per-call."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_error_violation(xt, yt, yhat, uhat, eps=1e-5):
    """RMSE, empirical coverage-violation rate, and mean uncertainty width
    for a (yhat, uhat) prediction against ground truth (xt, yt), where the
    predicted interval is [yhat - uhat, yhat + uhat]. `xt` is accepted (and
    unused) for parity with how this is called throughout the notebooks."""
    lb = yhat - uhat
    ub = yhat + uhat
    vio = 1 - np.bitwise_and(lb < (yt + eps), yt < (ub + eps)).sum() / yt.shape[0]
    error = np.sqrt(((yhat - yt) ** 2).mean())
    return error, vio, uhat.mean()
