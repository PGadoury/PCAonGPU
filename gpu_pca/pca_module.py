# Portions adapted from sirluk/pytorch_incremental_pca (MIT); see LICENSE.
from __future__ import annotations

import contextlib
import math
from numbers import Integral, Real
from typing import Optional

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
        n_components: Optional[int] = None,
        *,
        whiten: bool = False,
        device=None,
        copy: bool = True,
        batch_size: Optional[int] = None,
        svd_driver: Optional[str] = None,
        lowrank: bool = False,
        lowrank_q: Optional[int] = None,
        lowrank_niter: int = 4,
        lowrank_seed: Optional[int] = None,
        gram: bool = False,
        # New knobs
        stats_dtype: Optional[torch.dtype] = None,
        ensure_contiguous: bool = True,
        gram_eps: float = 1e-7,
        # Perf knobs
        allow_tf32: Optional[bool] = None,
        matmul_precision: Optional[
            str
        ] = None,  # "highest" | "high" | "medium" (torch>=2.0)
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
        self._x_aug_work: Optional[torch.Tensor] = None

    @staticmethod
    def _is_positive_integer(value) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool) and value > 0

    def _validate_parameters(self):
        if self.svd_driver not in (None, "gesvd", "gesvdj", "gesvda"):
            raise ValueError("Invalid svd_driver.")
        if self.svd_driver is not None and self.device.type != "cuda":
            raise ValueError("svd_driver requires a CUDA device.")
        if self.matmul_precision not in (None, "highest", "high", "medium"):
            raise ValueError("Invalid matmul_precision.")
        if self.n_components is not None and not self._is_positive_integer(
            self.n_components
        ):
            raise ValueError("n_components must be a positive integer or None.")
        if self.batch_size is not None and not self._is_positive_integer(
            self.batch_size
        ):
            raise ValueError("batch_size must be a positive integer or None.")
        if self.lowrank_q is not None and not self._is_positive_integer(self.lowrank_q):
            raise ValueError("lowrank_q must be a positive integer or None.")
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
        # Scoped TF32 / matmul precision toggles; restored afterwards.
        old_tf32 = None
        old_cudnn_tf32 = None
        old_prec = None
        changed_tf32 = self.allow_tf32 is not None and torch.cuda.is_available()
        changed_prec = self.matmul_precision is not None and hasattr(
            torch, "set_float32_matmul_precision"
        )

        try:
            if changed_tf32:
                old_tf32 = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = bool(self.allow_tf32)
                # cudnn TF32 can matter for some ops; harmless to mirror
                if hasattr(torch.backends, "cudnn") and hasattr(
                    torch.backends.cudnn, "allow_tf32"
                ):
                    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
                    torch.backends.cudnn.allow_tf32 = bool(self.allow_tf32)

            if changed_prec:
                # torch.get_float32_matmul_precision exists on modern PyTorch
                if hasattr(torch, "get_float32_matmul_precision"):
                    old_prec = torch.get_float32_matmul_precision()
                torch.set_float32_matmul_precision(self.matmul_precision)

            yield
        finally:
            if (
                changed_prec
                and old_prec is not None
                and hasattr(torch, "set_float32_matmul_precision")
            ):
                torch.set_float32_matmul_precision(old_prec)
            if changed_tf32 and old_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = old_tf32
            if (
                changed_tf32
                and old_cudnn_tf32 is not None
                and hasattr(torch.backends, "cudnn")
            ):
                torch.backends.cudnn.allow_tf32 = old_cudnn_tf32

    def _svd_fn_full(self, X):
        return torch.linalg.svd(X, full_matrices=False, driver=self.svd_driver)

    def _svd_fn_lowrank(self, X):
        q = self.lowrank_q
        if q is None:
            q = self.n_components_ * 2
        q = min(q, min(X.shape))
        if q < self.n_components_:
            raise ValueError("lowrank_q must be >= n_components_.")

        seed_enabled = self.lowrank_seed is not None
        with torch.random.fork_rng(enabled=seed_enabled):
            if seed_enabled:
                torch.manual_seed(self.lowrank_seed)
            U, S, V = torch.svd_lowrank(X, q=q, niter=self.lowrank_niter)
            return U, S, V.mH

    def _svd_fn_gram_topk(self, X):
        """
        Wide-matrix fast path: G = X @ X.T then eigh(G), recover Vt.
        Avoids flipping full eigensystem; slices only top-k.
        Also fuses invS scaling into the small (k x m) factor before GEMM.
        """
        m, D = X.shape
        if m > D:
            U, S, Vt = self._svd_fn_full(X)
            return U, S, Vt, None, None

        k = min(self.n_components_, m)

        # G is (m, m)
        G = X @ X.mH
        max_abs_diagonal = G.diagonal().real.abs().max()
        loading = torch.maximum(
            max_abs_diagonal.new_tensor(float(self.gram_eps) ** 2),
            torch.finfo(G.dtype).eps * max(1, m) * max_abs_diagonal,
        )
        G.diagonal().add_(loading)

        try:
            _evals, evecs = torch.linalg.eigh(G)  # ascending
        except torch.linalg.LinAlgError:
            U, S, Vt = self._svd_fn_full(X)
            return U, S, Vt, None, None

        # Take largest-k (from the end) then flip just those to descending
        U_k = evecs[:, -k:].flip(1)  # (m, k)

        # The diagonal loading is only for the eigensolver. Recover the actual,
        # unshifted spectrum and right singular vectors from the original X.
        Y = U_k.mH @ X
        S_k = torch.linalg.vector_norm(Y, dim=1)
        if (not bool(torch.isfinite(S_k).all())) or bool((S_k <= self.gram_eps).any()):
            U, S, Vt = self._svd_fn_full(X)
            return U, S, Vt, None, None
        Vt_k = Y / S_k[:, None]

        tail_count = m - k
        if tail_count > 0:
            tail_ss = (X.abs().square().sum() - S_k.square().sum()).clamp(min=0)
        else:
            tail_ss = torch.zeros((), device=X.device, dtype=X.dtype)

        return U_k, S_k, Vt_k, tail_ss, tail_count

    @staticmethod
    def _input_tensor(X):
        X = torch.as_tensor(X)
        if X.ndim != 2 or min(X.shape) == 0:
            raise ValueError("X must be a nonempty 2D input.")
        return X

    def _prepare(self, X, dtype=None, check_input=True):
        X = self._input_tensor(X)
        if dtype is None:
            dtype = (
                self.components_.dtype
                if hasattr(self, "components_")
                else X.dtype
                if X.dtype
                in (torch.float32, torch.float64, torch.complex64, torch.complex128)
                else torch.float32
            )
        if dtype not in (
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ):
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

    @staticmethod
    def _svd_flip(u, v, u_based_decision=False):
        # For complex data rotate phases while preserving U @ diag(S) @ Vh.
        rows = torch.arange(v.shape[0], device=v.device)
        pivots = v[rows, v.abs().argmax(dim=1)]
        phase = torch.sgn(pivots)
        phase = torch.where(phase.abs() == 0, torch.ones_like(phase), phase)
        return u * phase, v * phase.conj()[:, None]

    def _get_x_aug_work(self, rows, features, device, dtype):
        work = self._x_aug_work
        if (
            work is None
            or work.shape[0] < rows
            or work.shape[1] != features
            or work.device != device
            or work.dtype != dtype
        ):
            self._x_aug_work = torch.empty((rows, features), device=device, dtype=dtype)
        return self._x_aug_work[:rows]

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
        if first:
            mean, ss = batch_mean, batch_ss
        else:
            delta = batch_mean - self.mean_.to(stat_dtype)
            mean = self.mean_.to(stat_dtype) + delta * (mass / total)
            ss = (
                self.var_.to(real_dtype) * old_mass
                + batch_ss
                + delta.abs().square() * (old_mass * mass / total)
            )
        center = batch_mean.to(X.dtype)
        if first:
            matrix = X - center if self.copy else X.sub_(center)
            matrix = matrix * w.sqrt().to(X.real.dtype)[:, None]
        else:
            matrix = self._get_x_aug_work(k + rows + 1, features, X.device, X.dtype)
            torch.mul(self.components_, self.singular_values_[:, None], out=matrix[:k])
            torch.sub(X, center, out=matrix[k : k + rows])
            matrix[k : k + rows].mul_(w.sqrt().to(X.real.dtype)[:, None])
            matrix[-1].copy_(
                (self.mean_ - batch_mean) * math.sqrt(old_mass * mass / total)
            )
        # Backends use the effective rank, including when inferred from the first batch.
        self.n_components_ = k
        tail_ss = tail_count = None
        with self._matmul_context():
            if self.gram:
                U, S, Vh, tail_ss, tail_count = self._svd_fn_gram_topk(matrix)
            elif self.lowrank:
                U, S, Vh = self._svd_fn_lowrank(matrix)
            else:
                U, S, Vh = self._svd_fn_full(matrix)
        if self.deterministic_flip:
            U, Vh = self._svd_flip(U, Vh)
        # Reliability-weighted unbiased covariance; equals n - 1 for unit weights.
        dof = max(0.0, total - total_square / total)
        S2 = S.square()
        variance = S2 / dof if dof > 0 else torch.zeros_like(S2)
        energy = ss.sum()
        ratio = S2 / energy if energy > 0 else torch.zeros_like(S2)
        self.components_ = Vh[:k].clone()
        self.singular_values_ = S[:k].clone()
        self.mean_ = mean
        self.var_ = ss / total
        self.n_features_ = features
        self.n_samples_seen_ = rows + (0 if first else self.n_samples_seen_)
        self.weight_sum_ = total
        self.weight_square_sum_ = total_square
        self.explained_variance_ = variance[:k]
        self.explained_variance_ratio_ = ratio[:k]
        self.mean_proj_ = self.mean_.to(X.dtype) @ self.components_.mH
        discarded = min(matrix.shape) - k
        if discarded > 0 and dof > 0:
            if tail_ss is not None:
                residual = tail_ss
            elif self.lowrank:
                residual = (matrix.abs().square().sum() - S2[:k].sum()).clamp(min=0)
            else:
                residual = S2[k:].sum()
            self.noise_variance_ = residual / (dof * discarded)
        else:
            self.noise_variance_ = S.new_zeros(())
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
