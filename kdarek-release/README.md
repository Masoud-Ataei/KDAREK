# K-DAREK

K-DAREK (KAN-based Distance-Aware error-bound REgression with Kolmogorov-Arnold
networks) is a Lipschitz-bounded regression model that pairs an MLP front-end
with a spline-based (`DAREK`) uncertainty-aware KAN block, giving predictions
with a provable, distance-aware uncertainty band rather than a heuristic one.

This repo is a clean, minimal extraction of the K-DAREK experiments, meant to be
reproducible on its own — no pretrained checkpoints are shipped; every notebook
trains from scratch (a few seconds to ~1 minute per model on CPU).

## Install

```bash
pip install -r requirements.txt
```

Then, from the repo root, `import kdarek` works directly (or `pip install -e .`
if you'd rather not rely on `PYTHONPATH`).

## Package: `kdarek/`

```python
from kdarek import KDAREK, DAREK, Dataset, count_parameters, set_seed, check_error_violation

model = KDAREK(mlp_width=[1, 5], kan_width=[5, 1], kan_grid=8, kan_k=3,
                kan_base_fun='silu', device='cpu', kan_extend=True)
dataset = Dataset(fx=lambda x: 10 * np.cos(x), n=50, fix=True, seed=12)
model.fit(dataset, lr=0.1, steps=1000, nonfixknot=True, rand_method='Kmean')
yhat, u = model.predict(dataset['test_input'], L_k=10, L_1=10)  # yhat +/- u
```

- **`KDAREK`** (also importable as `K2DAREK`) — the full MLP+KAN model.
- **`DAREK`** — the spline-only block K-DAREK's KAN half is built from; usable
  standalone as a smaller baseline.
- **`Dataset`** — the toy 1D regression dataset generator used throughout these
  notebooks (fixed/evenly-spaced or random x, optional `noise=` for iid Gaussian
  label noise).
- **`lipschitz_share.py` / `error_share.py`** — Lipschitz-budget and error-share
  allocation strategies (`Equal_Lipschitz`, `Heuristic_Lipschitz`,
  `DataDriven_Lipschitz`, `NonOptimal_WorstCase_Lipschitz`, `SHAP_Error_Share`, …)
  used by the ablation study.
- **`count_parameters`, `set_seed`, `check_error_violation`** — small utilities
  shared across the notebooks.

## Notebooks

1. **`01_cos_function_extend.ipynb`** — trains K-DAREK on `10·cos(x)` with
   `kan_extend=True` and `kan_extend=False` side by side and compares the fits,
   then builds a K-DAREK instance with a specific, fixed **200-parameter**
   budget (verified via `count_parameters`, not assumed).
2. **`02_ablation.ipynb`** — an architecture ablation isolating what actually
   drives K-DAREK's behavior: MLP block present/absent, spectral normalization
   on/off, 5 knot-selection strategies, spline capacity/order, and 6
   Lipschitz-budget/error-share allocation strategies (run on both the
   spline-only baseline and the full model).
3. **`03_crown_bounds.ipynb`** — formal verification bounds via
   [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)/CROWN on
   K-DAREK's forward pass, compared against K-DAREK's own Lipschitz bound.
   **Needs its own bundled dependencies** (`notebooks/03_crown_deps/`) — see
   below.
4. **`04_knot_selection.ipynb`** — compares 6 knot-selection strategies
   (`Kmean`, `random`, `LHS`, `gw_kmean`, `igw_kmean`, `chebyshev`): a results
   table, an overlay of all fits, and a per-strategy plot of where each one
   actually places its knots.

### Why notebook 3 has its own dependency copies

`notebooks/03_crown_deps/` bundles its own `K2DAREK.py`/`DAREK.py`/`KKAN.py`
plus a minimal, patched copy of [pykan](https://github.com/KindXiaoming/pykan)
(`pykan/kan/` only — docs/tutorials/etc. stripped out), instead of using the
`kdarek` package the other three notebooks share. Two reasons:

- That variant's `fit()` has a fuller, original KAN-style API the CROWN patches
  were written and verified against — the `kdarek` package's `fit()` is a
  simplified, Adam-only variant with a different signature.
- The custom `auto_LiRPA` bound operator (`custom::KANSpline`) notebook 3
  registers is written against this exact pykan copy's `coef2curve`/`KANLayer`
  internals.

Install `auto-LiRPA==0.3` (commented out in `requirements.txt`, since it's only
needed for this one notebook) before running it.

## License

MIT — see `LICENSE`.
