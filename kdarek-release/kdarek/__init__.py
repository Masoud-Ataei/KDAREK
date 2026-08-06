"""kdarek -- Lipschitz-bounded, uncertainty-aware KAN regression.

Public API:
    DAREK      -- spline-only baseline (Distance-Aware error-bound KAN)
    KDAREK     -- full model: an MLP front-end feeding a DAREK spline block,
                  with a Lipschitz-bounded uncertainty band. Also importable
                  as K2DAREK, matching how it's referred to in the notebooks
                  and in the paper.
    Dataset    -- toy 1D regression dataset generator used throughout the
                  notebooks (fixed/evenly-spaced or random x, optional
                  Gaussian label noise via `noise=`)
    MLP, Ensemble_KAN, GPs -- baseline models used for comparison
    count_parameters, set_seed, check_error_violation -- small utilities
"""

from .darek import DAREK
from .k2darek import KDAREK
from .kkan import Dataset, MLP, Ensemble_KAN, GPs
from .utils import count_parameters, set_seed, check_error_violation

# KDAREK is the class name in the source; K2DAREK is how it's referred to
# everywhere else (notebooks, README, paper) -- same object, both names work.
K2DAREK = KDAREK

__all__ = [
    "DAREK", "KDAREK", "K2DAREK",
    "Dataset", "MLP", "Ensemble_KAN", "GPs",
    "count_parameters", "set_seed", "check_error_violation",
]
