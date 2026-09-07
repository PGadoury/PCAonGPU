# PCAonGPU

Incremental principal component analysis using PyTorch, with CPU/CUDA execution,
real or complex data, optional sample weights, and three decomposition backends.

```bash
pip install PCAonGPU
# From a source checkout, including test dependencies:
pip install -e '.[test]'
```

```python
import torch
from gpu_pca import IncrementalPCAonGPU

X = torch.randn(10000, 128)  # Can remain in CPU memory.
pca = IncrementalPCAonGPU(n_components=16, batch_size=512)
pca.fit(X)                 # Transfers each batch to the selected device.
Z = pca.transform(X)       # Output resides on that device.
```

`device=None` automatically selects CUDA when available; use `device="cpu"` or
`device="cuda:1"` to select explicitly. `fit()` resets learned state. For streaming,
call `partial_fit(batch)` repeatedly. The first positive-weight batch must have at
least `n_components` rows; later batches may be smaller. With `n_components=None`,
the component count is inferred once from the first positive-weight batch.

`fit()` bounds input transfer and decomposition workspace by batch size. The
original dataset still needs host storage; use `partial_fit()` for external data
streams. `transform()` transfers its entire input and returns the entire output;
call it on batches to bound transformation memory.

## Weights and statistics

```python
weights = torch.rand(len(X))
pca.fit(X, X_weights=weights, dtype=torch.float64)
Z = pca.transform(X)  # Automatically uses the fitted dtype.
```

Weights are **relative importance weights**, not replication counts. They must be
finite and nonnegative, and may be a scalar, a vector of length `n_samples`, or an
`(n_samples, 1)` column. Scalars are expanded before batching. A zero-total-weight
`partial_fit()` is a no-op; an entirely zero-weight `fit()` raises `ValueError`.

For total weight `W = sum(w)` and `Q = sum(w**2)`, `var_` is the weighted population
variance, while `explained_variance_` and `noise_variance_` use the unbiased
reliability-weight denominator `W - Q / W`. This becomes `n - 1` for unit weights
and makes variances invariant to a common rescaling of all weights. If the
denominator is zero (one effective observation), explained and noise variances
are zero. `singular_values_` measures unnormalized weighted scatter and therefore
scales by `sqrt(c)` when all weights are multiplied by `c`.

`n_samples_seen_` counts rows from positive-total-weight batches, including rows
with individual zero weights. `weight_sum_` and `weight_square_sum_` track `W` and
`Q`. `mean_`, `var_`, `components_`, `explained_variance_ratio_`, `n_features_`,
`n_components_` and, after `fit()`, `batch_size_` are also available. Constant data
has zero explained-variance ratios.

## Dtypes, copying and whitening

`dtype=None` preserves float32, float64, complex64 and complex128 input on first
fit; other real types convert to float32. Later updates and transforms use the
fitted dtype. Explicit dtype changes require a fresh `fit()`. Complex inputs use
magnitude-squared statistics and conjugate projections; complex-to-real casts are
rejected. `stats_dtype=torch.float64` enables higher-precision accumulation.

`copy=True` preserves input. `copy=False` permits in-place first-batch centering
when no device/dtype conversion is needed. `transform()` preserves input and uses
a cached mean projection. `fit_transform()` preserves the original data for its
projection even with `copy=False`.

`whiten=True` scales projected coordinates by the square root of the learned
explained variance. Zero-variance scales are clipped to machine epsilon.
`check_input=False` skips checking input finiteness; shape, dtype and weight
validation remain enabled. Existing positional `check_input` calls still work;
`X_weights` and `dtype` are keyword-only.

## Backends

- Default: exact `torch.linalg.svd` on each augmented batch.
- `gram=True`: solves a smaller Gram eigenproblem for wide matrices, with
  scale-aware diagonal loading and full-SVD fallback for tall, degenerate or
  failed eigenproblems. Reported singular values exclude the loading. Forming a
  Gram matrix squares the condition number, so use full SVD for sensitive data.
- `lowrank=True`: randomized SVD. Tune `lowrank_q` (defaults to twice the retained
  rank, capped by matrix dimensions), `lowrank_niter`, and `lowrank_seed`.
  Residual noise includes energy outside the approximate retained subspace.

Gram and low-rank modes are mutually exclusive. `svd_driver` selects a CUDA SVD
driver; `allow_tf32` and `matmul_precision` temporarily control matrix products.

Even full-SVD incremental PCA is an approximation to a single PCA on all data
when intermediate updates discard components. Results can depend on batch size
and order. Exact covariance equivalence is expected when all components are kept.

## Tests and benchmarks

```bash
python -m pytest -q
python -m tests.benchmark_gpu_pca --samples 5000 --features 256 --components 16
```

Tests use seeded CPU data, exercise CUDA when available, and compare against
scikit-learn, direct weighted covariance, and full-SVD references. Benchmarks
synchronize CUDA and report timings separately from numerical tests. Performance
depends on matrix shape, retained rank, backend and device.

The backend and workspace implementation incorporates MIT-licensed code from
[sirluk/pytorch_incremental_pca](https://github.com/sirluk/pytorch_incremental_pca/tree/d962f0439818b0d59286ad218a5460bcdbd48e7d).
Its copyright and license are preserved in `LICENSE`. Sample-weight support builds
on [PR #6](https://github.com/dnhkng/PCAonGPU/pull/6).
