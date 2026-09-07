# Portions adapted from sirluk/pytorch_incremental_pca (MIT); see LICENSE.
from __future__ import annotations

import contextlib
import math
from numbers import Integral, Real

import torch


class IncrementalPCAonGPU:
    """Incremental PCA with optional sample weights and CPU/CUDA execution.

    Full SVD is the default. ``gram=True`` selects a wide-matrix Gram solver
    with full-SVD fallback; ``lowrank=True`` selects randomized SVD, controlled
    by ``lowrank_q``, ``lowrank_niter`` and ``lowrank_seed``. These modes are
    mutually exclusive. Gram matrices square the condition number.

    ``device=None`` selects CUDA when available, otherwise CPU. ``fit`` transfers
    one input batch at a time. ``copy=False`` permits first-batch centering in
    place; ``transform`` always preserves input. ``whiten=True`` divides projected
    coordinates by the square root of the learned explained variance.

    ``stats_dtype`` selects float32/float64 accumulation (including the matching
    complex dtype for complex means). By default CPU statistics use float64 and
    CUDA statistics retain the input precision. ``svd_driver`` is CUDA-only.
    ``allow_tf32`` and ``matmul_precision`` temporarily control matrix products.

    See README.md for weight semantics, dtype conversion and approximation limits.
    """

    def __init__(
        self,
        n_components: int | None = None,
        *,
        whiten: bool = False,
        device=None,
        copy: bool = True,
        batch_size: int | None = None,
        svd_driver: str | None = None,
        lowrank: bool = False,
        lowrank_q: int | None = None,
        lowrank_niter: int = 4,
        lowrank_seed: int | None = None,
        gram: bool = False,
        stats_dtype: torch.dtype | None = None,
        ensure_contiguous: bool = True,
        gram_eps: float = 1e-7,
        allow_tf32: bool | None = None,
        matmul_precision: str | None = None,
        deterministic_flip: bool = True,
    ):
        self.whiten = whiten
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.n_components = n_components
        self.copy = copy
        self.batch_size = batch_size
        self.svd_driver = svd_driver

        self.lowrank = lowrank
        self.lowrank_q = lowrank_q
        self.lowrank_niter = lowrank_niter
        self.lowrank_seed = lowrank_seed

        self.gram = gram
        self.stats_dtype = stats_dtype
        self.ensure_contiguous = ensure_contiguous
        self.gram_eps = gram_eps

        self.allow_tf32 = allow_tf32
        self.matmul_precision = matmul_precision
        self.deterministic_flip = deterministic_flip

        self._reset_fit_state()
        self._validate_parameters()

    def _reset_fit_state(self):
        """Remove all state learned from earlier calls to fit."""
        learned_attributes = (
            "components_",
            "singular_values_",
            "mean_",
            "var_",
            "explained_variance_",
            "explained_variance_ratio_",
            "noise_variance_",
            "mean_proj_",
            "n_components_",
            "batch_size_",
            "n_features_",
            "n_samples_seen_",
            "weight_sum_",
            "weight_square_sum_",
        )
        for attribute in learned_attributes:
            self.__dict__.pop(attribute, None)

        # Workspace for the augmented matrix; it must not survive a refit on a
        # different shape, device, or dtype.
        self._x_aug_work: torch.Tensor | None = None

    def _validate_parameters(self):
        for name in ("n_components", "batch_size", "lowrank_q"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, Integral) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None.")
        if self.svd_driver not in (None, "gesvd", "gesvdj", "gesvda"):
            raise ValueError("Invalid svd_driver.")
        if self.svd_driver is not None and self.device.type != "cuda":
            raise ValueError("svd_driver requires a CUDA device.")
        if self.matmul_precision not in (None, "highest", "high", "medium"):
            raise ValueError("Invalid matmul_precision.")
        if (
            not isinstance(self.lowrank_niter, Integral)
            or isinstance(self.lowrank_niter, bool)
            or self.lowrank_niter < 0
        ):
            raise ValueError("lowrank_niter must be a nonnegative integer.")
        if (
            not isinstance(self.gram_eps, Real)
            or isinstance(self.gram_eps, bool)
            or not math.isfinite(float(self.gram_eps))
            or self.gram_eps <= 0
        ):
            raise ValueError("gram_eps must be finite and strictly positive.")
        if self.stats_dtype not in (None, torch.float32, torch.float64):
            raise ValueError(
                "stats_dtype must be torch.float32, torch.float64, or None."
            )
        if self.lowrank and self.gram:
            raise ValueError(
                "lowrank and gram are mutually exclusive. Set only one to True."
            )
        if (
            self.lowrank_q is not None
            and self.n_components is not None
            and self.lowrank_q < self.n_components
        ):
            raise ValueError("lowrank_q must be >= n_components.")

    @contextlib.contextmanager
    def _matmul_context(self):
        """Temporarily apply the requested matrix multiplication precision."""
        if self.matmul_precision is None and self.allow_tf32 is None:
            yield
            return
        precision = torch.get_float32_matmul_precision()
        tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            if self.allow_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            if self.matmul_precision is not None:
                torch.set_float32_matmul_precision(self.matmul_precision)
            yield
        finally:
            torch.set_float32_matmul_precision(precision)
            torch.backends.cuda.matmul.allow_tf32 = tf32

    def _svd_fn_full(self, X):
        _, S, Vh = torch.linalg.svd(X, full_matrices=False, driver=self.svd_driver)
        return S, Vh

    def _svd_lowrank(self, X):
        q = min(self.lowrank_q or 2 * self.n_components_, min(X.shape))
        if q < self.n_components_:
            raise ValueError("lowrank_q must be >= n_components_.")
        with torch.random.fork_rng(enabled=self.lowrank_seed is not None):
            if self.lowrank_seed is not None:
                torch.manual_seed(self.lowrank_seed)
            _, S, V = torch.svd_lowrank(X, q=q, niter=self.lowrank_niter)
        return S, V.mH

    def _svd_gram(self, X):
        """Return leading singular pairs, or None when full SVD is safer."""
        rows, features = X.shape
        if rows > features:
            return None
        gram = X @ X.mH
        loading = torch.finfo(X.dtype).eps * rows * gram.diagonal().real.abs().max()
        gram.diagonal().add_(loading.clamp_min(self.gram_eps**2))
        try:
            _, vectors = torch.linalg.eigh(gram)
        except torch.linalg.LinAlgError:
            return None
        # Recover values from the original matrix, excluding diagonal loading.
        projected = vectors[:, -self.n_components_ :].flip(1).mH @ X
        S = torch.linalg.vector_norm(projected, dim=1)
        if not torch.isfinite(S).all() or (S <= self.gram_eps).any():
            return None
        return S, projected / S[:, None]

    def _decompose(self, X):
        """Return retained singular values, component rows, and discarded energy."""
        approximate = False
        with self._matmul_context():
            if self.lowrank:
                result = self._svd_lowrank(X)
                approximate = True
            elif self.gram:
                result = self._svd_gram(X)
                approximate = result is not None
            else:
                result = None
            S, Vh = self._svd_fn_full(X) if result is None else result
        k = self.n_components_
        residual = (
            (X.abs().square().sum() - S[:k].square().sum()).clamp_min(0)
            if approximate
            else S[k:].square().sum()
        )
        S, Vh = S[:k].clone(), Vh[:k].clone()
        if self.deterministic_flip:
            pivots = Vh.gather(1, Vh.abs().argmax(dim=1, keepdim=True))
            # Fix complex phase as well as real sign; left vectors are unused.
            Vh *= torch.sgn(pivots).conj()
        return S, Vh, residual

    @staticmethod
    def _input_tensor(X):
        X = torch.as_tensor(X)
        if X.ndim != 2 or min(X.shape) == 0:
            raise ValueError("X must be a nonempty 2D input.")
        return X

    def _prepare(self, X, dtype=None, check_input=True):
        X = self._input_tensor(X)
        valid_dtypes = (torch.float32, torch.float64, torch.complex64, torch.complex128)
        if dtype is None:
            if hasattr(self, "components_"):
                dtype = self.components_.dtype
            else:
                dtype = X.dtype if X.dtype in valid_dtypes else torch.float32
        if dtype not in valid_dtypes:
            raise ValueError(
                "dtype must be float32, float64, complex64, or complex128."
            )
        if X.is_complex() and not dtype.is_complex:
            raise ValueError("Cannot cast complex input to a real dtype.")
        if hasattr(self, "components_"):
            if X.shape[1] != self.n_features_:
                raise ValueError("Number of features does not match the fitted model.")
            if dtype != self.components_.dtype:
                raise ValueError(
                    "dtype must match the fitted model; call fit to change it."
                )
        X = X.to(device=self.device, dtype=dtype)
        if check_input and not bool(torch.isfinite(X).all()):
            raise ValueError("X must contain only finite values.")
        return X.contiguous() if self.ensure_contiguous else X

    @staticmethod
    def _weights(weights, n):
        if weights is None:
            return None
        w = torch.as_tensor(weights)
        if w.is_complex() or w.ndim > 2 or (w.ndim == 2 and w.shape[1] != 1):
            raise ValueError(
                "Weights must be real scalars, vectors, or column vectors."
            )
        w = w.reshape(-1)
        if w.numel() == 1:
            w = w.expand(n)
        if w.numel() != n:
            raise ValueError("Weights must have one entry per sample.")
        if not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
            raise ValueError("Weights must be finite and nonnegative.")
        return w

    def _augmented_matrix(self, X, weights, batch_mean, mass):
        """Combine retained scatter, the centered batch, and the mean correction."""
        center = batch_mean.to(X.dtype)
        scale = weights.sqrt().to(X.real.dtype)[:, None]
        if not hasattr(self, "components_"):
            centered = X - center if self.copy else X.sub_(center)
            return centered * scale

        k = self.n_components_
        rows, features = X.shape
        size = k + rows + 1
        work = self._x_aug_work
        if work is None or work.shape[0] < size:
            # Feature count and dtype are fixed during partial_fit; fit clears this.
            self._x_aug_work = X.new_empty(size, features)
        matrix = self._x_aug_work[:size]
        torch.mul(self.components_, self.singular_values_[:, None], out=matrix[:k])
        torch.sub(X, center, out=matrix[k:-1])
        matrix[k:-1].mul_(scale)
        correction = math.sqrt(self.weight_sum_ * mass / (self.weight_sum_ + mass))
        matrix[-1].copy_((self.mean_ - batch_mean) * correction)
        return matrix

    def _merge_statistics(self, mean, scatter, mass):
        """Merge population scatter using the difference between batch means."""
        if not hasattr(self, "components_"):
            return mean, scatter
        previous_mean = self.mean_.to(mean.dtype)
        previous_scatter = self.var_.to(scatter.dtype) * self.weight_sum_
        total = self.weight_sum_ + mass
        delta = mean - previous_mean
        mean = previous_mean + delta * (mass / total)
        correction = delta.abs().square() * (self.weight_sum_ * mass / total)
        return mean, previous_scatter + scatter + correction

    @torch.inference_mode()
    def fit(self, X, check_input=True, *, X_weights=None, dtype=None):
        """Reset and fit. CPU data and weights are transferred one batch at a time.

        Weights are relative importance weights; see README for variance normalization.
        dtype=None preserves supported input dtypes. check_input=False skips finite-X
        checking only; shape, weight and dtype invariants are always checked.
        """
        self._reset_fit_state()
        self._validate_parameters()
        X = self._input_tensor(X)
        weights = self._weights(X_weights, X.shape[0])
        if weights is not None and not bool((weights > 0).any()):
            raise ValueError("At least one sample must have positive weight.")
        self.batch_size_ = self.batch_size or 5 * X.shape[1]
        if self.gram and self.batch_size is None and self.n_components is not None:
            self.batch_size_ = max(
                self.n_components, X.shape[1] - self.n_components - 1
            )
        if self.n_components is not None and self.batch_size_ < self.n_components:
            raise ValueError("batch_size must be >= n_components.")
        for start in range(0, X.shape[0], self.batch_size_):
            batch = slice(start, start + self.batch_size_)
            self.partial_fit(
                X[batch],
                check_input=check_input,
                X_weights=None if weights is None else weights[batch],
                dtype=dtype,
            )
        return self

    @torch.inference_mode()
    def partial_fit(self, X, check_input=True, *, X_weights=None, dtype=None):
        """Update from one batch. A zero-total-weight batch is a no-op.

        The first positive-weight batch needs at least n_components rows.
        Subsequent batches may be smaller. Existing state fixes feature count and dtype.
        """
        self._validate_parameters()
        X = self._prepare(X, dtype, check_input)
        rows, features = X.shape
        w = self._weights(X_weights, rows)
        real_dtype = self.stats_dtype or X.real.dtype
        if self.stats_dtype is None and not X.is_cuda:
            real_dtype = torch.float64
        stat_dtype = (
            {torch.float32: torch.complex64, torch.float64: torch.complex128}[
                real_dtype
            ]
            if X.is_complex()
            else real_dtype
        )
        xs = X.to(stat_dtype)
        w = (
            torch.ones(rows, device=X.device, dtype=real_dtype)
            if w is None
            else w.to(device=X.device, dtype=real_dtype)
        )
        mass = w.sum().item()
        square_mass = w.square().sum().item()
        if not math.isfinite(mass) or not math.isfinite(square_mass):
            raise ValueError("Weight totals overflow the statistics dtype.")
        if mass == 0:
            return self
        first = not hasattr(self, "components_")
        k = (self.n_components or min(rows, features)) if first else self.n_components_
        if k > features or (first and k > rows):
            raise ValueError(
                "n_components must not exceed features or first batch rows."
            )
        batch_mean = (xs * w[:, None]).sum(0) / mass
        batch_ss = ((xs - batch_mean).abs().square() * w[:, None]).sum(0)
        old_mass = 0 if first else self.weight_sum_
        total = old_mass + mass
        total_square = square_mass + (0 if first else self.weight_square_sum_)
        mean, ss = self._merge_statistics(batch_mean, batch_ss, mass)
        matrix = self._augmented_matrix(X, w, batch_mean, mass)
        self.n_components_ = k
        S, components, residual = self._decompose(matrix)
        # Reliability-weighted unbiased covariance; equals n - 1 for unit weights.
        dof = max(0.0, total - total_square / total)
        S2 = S.square()
        variance = S2 / dof if dof > 0 else torch.zeros_like(S2)
        energy = ss.sum()
        ratio = S2 / energy if energy > 0 else torch.zeros_like(S2)
        self.components_ = components
        self.singular_values_ = S
        self.mean_ = mean
        self.var_ = ss / total
        self.n_features_ = features
        self.n_samples_seen_ = rows + (0 if first else self.n_samples_seen_)
        self.weight_sum_ = total
        self.weight_square_sum_ = total_square
        self.explained_variance_ = variance
        self.explained_variance_ratio_ = ratio
        self.mean_proj_ = self.mean_.to(X.dtype) @ self.components_.mH
        discarded = min(matrix.shape) - k
        self.noise_variance_ = (
            residual / (dof * discarded)
            if discarded > 0 and dof > 0
            else S.new_zeros(())
        )
        return self

    @torch.inference_mode()
    def transform(self, X, check_input=True, *, dtype=None):
        """Project onto learned components, optionally whitening. Input is preserved."""
        if not hasattr(self, "components_"):
            raise ValueError("Model must be fitted before transforming data.")
        X = self._prepare(X, dtype, check_input)
        with self._matmul_context():
            result = X @ self.components_.mH - self.mean_proj_
        if self.whiten:
            scale = self.explained_variance_.sqrt()
            result = result / scale.clamp_min(torch.finfo(scale.dtype).eps)
        return result

    def fit_transform(self, X, check_input=True, *, X_weights=None, dtype=None):
        """Fit and project the original data, including when copy=False."""
        X = self._input_tensor(X)
        original = X.clone() if not self.copy else X
        self.fit(X, check_input=check_input, X_weights=X_weights, dtype=dtype)
        return self.transform(original, check_input=check_input, dtype=dtype)
