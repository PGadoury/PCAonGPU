import numpy as np
import pytest
import torch
from sklearn.decomposition import IncrementalPCA as SklearnIPCA

from gpu_pca.pca_module import IncrementalPCAonGPU


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_data(n_samples=50000, n_features=100, random_state=None):
    """Generate random Gaussian data."""
    rng = np.random.default_rng(random_state)
    return rng.standard_normal(size=(n_samples, n_features))


data1 = generate_data()
data2 = generate_data()

data1gpu = torch.tensor(data1, device=DEVICE)
data2gpu = torch.tensor(data2, device=DEVICE)


def _as_float64_tensor(A):
    return torch.tensor(np.asarray(A), dtype=torch.float64)


def assert_subspaces_close(A, B, atol=1e-2):
    """Assert that the row spaces of A and B (component matrices) coincide.

    For orthonormal row bases of the same k-dimensional subspace, A @ B.T is an
    orthogonal matrix whose singular values are all 1. This is invariant to sign
    flips and rotations within the subspace.
    """
    A = _as_float64_tensor(A)
    B = _as_float64_tensor(B)
    s = torch.linalg.svdvals(A @ B.T)
    max_deviation = (s - 1).abs().max().item()
    assert max_deviation < atol, f"Subspaces differ by {max_deviation:.2e} (> {atol})"


def _weighted_pca_components(X, w, k):
    """Closed-form weighted PCA components (rows), for reference."""
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    mu = np.average(X, axis=0, weights=w)
    C = ((X - mu).T * w) @ (X - mu) / w.sum()  # weighted covariance
    eigvals, eigvecs = np.linalg.eigh(C)
    return eigvecs[::-1][:k].T  # top-k eigenvectors as rows


def test_fit_method():
    sklearn_model = SklearnIPCA(n_components=5)
    our_model = IncrementalPCAonGPU(n_components=5)

    sklearn_model.fit(data1)
    our_model.fit(data1gpu)

    transformed_sklearn = torch.tensor(sklearn_model.transform(data1), dtype=torch.float64)
    transformed_our_model = _as_float64_tensor(our_model.transform(data1gpu))

    # Transforms may differ by a per-component sign; align them before comparing.
    signs = torch.sign((transformed_our_model * transformed_sklearn).sum(dim=0)).clamp(-1, 1)
    assert torch.allclose(transformed_our_model * signs, transformed_sklearn, atol=5e-2), (
        "fit() transform does not match sklearn IncrementalPCA"
    )

    # Running statistics must match the exact batch statistics.
    assert np.allclose(our_model.mean_.cpu().numpy(), data1.mean(axis=0), atol=1e-4)
    assert np.allclose(our_model.var_.cpu().numpy(), data1.var(axis=0), atol=1e-3)


def test_partial_fit_method():
    sklearn_model = SklearnIPCA(n_components=5)
    our_model = IncrementalPCAonGPU(n_components=5)

    sklearn_model.partial_fit(data1)
    sklearn_model.partial_fit(data2)

    our_model.partial_fit(data1gpu)
    our_model.partial_fit(data2gpu)

    transformed_sklearn = torch.tensor(sklearn_model.transform(data1), dtype=torch.float64)
    transformed_our_model = _as_float64_tensor(our_model.transform(data1gpu))

    signs = torch.sign((transformed_our_model * transformed_sklearn).sum(dim=0)).clamp(-1, 1)
    assert torch.allclose(transformed_our_model * signs, transformed_sklearn, atol=5e-2), (
        "partial_fit() transform does not match sklearn IncrementalPCA"
    )


def test_incremental_mean_and_var_unweighted():
    """Regression: the parallel variance update must recover exact statistics.

    With batches [0, 2] then [4], the overall population mean is 2 and the
    population variance of [0, 2, 4] is 8/3. The pre-fix formula gave 7/3.
    """
    m1 = IncrementalPCAonGPU._incremental_mean_and_var(
        torch.tensor([[0.0], [2.0]]), None, None, None, 0
    )
    assert torch.allclose(m1[0], torch.tensor([1.0]), atol=1e-6)
    assert torch.allclose(m1[1], torch.tensor([2.0]), atol=1e-6)

    m2 = IncrementalPCAonGPU._incremental_mean_and_var(
        torch.tensor([[4.0]]), None, m1[0], m1[1], m1[2]
    )
    assert torch.allclose(m2[0], torch.tensor([2.0]), atol=1e-6)
    assert torch.allclose(m2[1], torch.tensor([8.0 / 3.0]), atol=1e-5), (
        f"incremental variance is {m2[1].item():.6f}, expected 8/3"
    )


def test_incremental_mean_and_var_weighted():
    """Weighted running statistics must match closed-form weighted statistics."""
    X = torch.tensor([[0.0], [2.0], [4.0]])
    w = torch.tensor([1.0, 2.0, 1.0])

    m1 = IncrementalPCAonGPU._incremental_mean_and_var(X, w, None, None, 0)
    # weighted mean = (0 + 4 + 4) / 4 = 2; S = 4 + 0 + 4 = 8; var = 8/4 = 2
    assert torch.allclose(m1[0], torch.tensor([2.0]), atol=1e-6)
    assert torch.allclose(m1[1], torch.tensor([2.0]), atol=1e-5)

    m2 = IncrementalPCAonGPU._incremental_mean_and_var(
        torch.tensor([[6.0]]), torch.tensor([3.0]), m1[0], m1[1], m1[2]
    )
    # total n = 7, mean = (4*2 + 3*6)/7 = 26/7
    # S_total = 8 + 0 + (4*3/7)*(2-6)^2 = 8 + 96/7; var = S_total / 7
    expected_var = (8.0 + 96.0 / 7.0) / 7.0
    assert torch.allclose(m2[0], torch.tensor([26.0 / 7.0]), atol=1e-5)
    assert torch.allclose(m2[1], torch.tensor([expected_var]), atol=1e-4), (
        f"incremental weighted variance is {m2[1].item():.6f}, expected {expected_var:.6f}"
    )


def test_weighted_first_pass_matches_weighted_pca():
    """Regression: the first partial_fit must compute *weighted* PCA of the batch.

    Pre-fix, only centering was applied on the first pass, so the initial
    components were the unweighted PCA of the batch (the "sensitive to initial
    weights" behaviour).
    """
    w = torch.rand(data1gpu.shape[0], device="cuda") ** 2
    w = torch.maximum(w, 1 - w)

    model = IncrementalPCAonGPU(n_components=5)
    model.partial_fit(data1gpu, w)

    ref = _weighted_pca_components(data1, w.cpu().numpy(), k=5)
    assert_subspaces_close(model.components_.cpu().numpy(), ref, atol=1e-2)


def test_weighted_partial_fit_method():
    """Two complementary weightings of the same batch must reproduce unweighted PCA.

    Fitting with weights w and then with 1 - w gives every sample a total weight
    of exactly 1, so the final components must match an unweighted fit on the
    same data (up to numerical error).
    """
    w = torch.rand(data1gpu.shape[0], device="cuda") ** 2
    w = torch.maximum(w, 1 - w)

    model_unweighted = IncrementalPCAonGPU(n_components=5)
    model_weighted = IncrementalPCAonGPU(n_components=5)

    model_unweighted.partial_fit(data1gpu)

    model_weighted.partial_fit(data1gpu, w)
    model_weighted.partial_fit(data1gpu, 1 - w)

    assert_subspaces_close(
        model_unweighted.components_.cpu().numpy(),
        model_weighted.components_.cpu().numpy(),
        atol=5e-2,
    )


def test_weighted_fit_matches_closed_form():
    """Regression: fit() with weights must equal closed-form weighted PCA.

    The stacked-matrix trick requires rows to be scaled by sqrt(w) (not w), so
    that Q^T Q equals the weighted sum of squares about the running mean.
    """
    w = torch.rand(data1gpu.shape[0], device="cuda") ** 2
    w = torch.maximum(w, 1 - w)

    model = IncrementalPCAonGPU(n_components=5)
    model.fit(data1gpu, w)

    ref = _weighted_pca_components(data1, w.cpu().numpy(), k=5)
    assert_subspaces_close(model.components_.cpu().numpy(), ref, atol=1e-2)


def test_fit_transform_weights_default_and_dtype():
    """Regression: fit_transform(X) must work without weights and keep the dtype.

    Pre-fix, X_weights had no default (TypeError), and transform() re-cast X to
    the default float32, which broke non-float32 fits.
    """
    X = data1[:2000]  # numpy float64 input
    model = IncrementalPCAonGPU(n_components=5)
    out = model.fit_transform(X, dtype=torch.float64)

    assert out.dtype == torch.float64
    assert out.shape == (X.shape[0], 5)
    assert np.allclose(
        out.cpu().numpy(),
        model.transform(X, check_input=True, dtype=torch.float64).cpu().numpy(),
    )


def test_noise_variance_wide_data():
    """Regression: default n_components on wide data must not raise TypeError.

    Pre-fix, the noise-variance branch compared self.n_components (None by
    default) against n_samples.
    """
    X = data1[:50]  # 50 samples < 100 features
    model_default = IncrementalPCAonGPU()
    model_default.fit(X)
    assert model_default.noise_variance_ == 0.0

    model_k = IncrementalPCAonGPU(n_components=5)
    model_k.fit(data1[:200])
    assert model_k.noise_variance_ > 0.0


def _to_f64(A):
    """Return A as a float64 tensor on the active device."""
    return torch.tensor(np.asarray(A), dtype=torch.float64, device=DEVICE)


def test_weighted_multi_batch_partial_fit_matches_closed_form():
    """Weighted partial_fit across >=3 batches must match closed-form weighted PCA.

    Feeding one dataset in chunks with matching weight slices exercises the merge
    path more than twice and must reproduce a single weighted PCA of all samples.
    """
    rng = np.random.default_rng(0)
    X = _to_f64(data1[:4000])
    w = torch.tensor(rng.random(X.shape[0]) ** 2, dtype=torch.float64, device=DEVICE)

    model = IncrementalPCAonGPU(n_components=5)
    n_chunks = 7
    for Xc, wc in zip(torch.chunk(X, n_chunks), torch.chunk(w, n_chunks)):
        model.partial_fit(Xc, wc)

    ref = _weighted_pca_components(X.cpu().numpy(), w.cpu().numpy(), k=5)
    assert_subspaces_close(model.components_.cpu().numpy(), ref, atol=1e-2)


def test_uniform_weights_match_unweighted():
    """Uniform weights must reproduce the unweighted result exactly.

    With every weight equal to a constant c, normalization cancels and weighted
    PCA reduces to ordinary PCA; this catches any residual sqrt/normalization slip.
    """
    X = _to_f64(data1[:2000])
    w = torch.full((X.shape[0],), 3.7, dtype=torch.float64, device=DEVICE)

    model_unweighted = IncrementalPCAonGPU(n_components=5).fit(X)
    model_weighted = IncrementalPCAonGPU(n_components=5).fit(X, w)

    assert_subspaces_close(
        model_unweighted.components_.cpu().numpy(),
        model_weighted.components_.cpu().numpy(),
        atol=1e-3,
    )


def test_batch_size_invariance_weighted():
    """Same weights in one batch vs many batches must give the same components."""
    rng = np.random.default_rng(1)
    X = _to_f64(data1[:4000])
    w = torch.tensor(rng.random(X.shape[0]), dtype=torch.float64, device=DEVICE)

    model_single = IncrementalPCAonGPU(n_components=5).fit(X, w)

    model_chunked = IncrementalPCAonGPU(n_components=5, batch_size=250)
    model_chunked.fit(X, w)

    assert_subspaces_close(
        model_single.components_.cpu().numpy(),
        model_chunked.components_.cpu().numpy(),
        atol=1e-3,
    )


def test_weighted_mean_and_var_attributes():
    """The estimator's mean_/var_ must match closed-form weighted statistics."""
    rng = np.random.default_rng(2)
    X = data1[:500]
    w = rng.random(X.shape[0]) ** 2

    model = IncrementalPCAonGPU(n_components=3, batch_size=97)
    model.fit(X, torch.tensor(w, dtype=torch.float64, device=DEVICE), dtype=torch.float64)

    mu = np.average(X, axis=0, weights=w)
    var = np.average((X - mu) ** 2, axis=0, weights=w)
    assert np.allclose(model.mean_.cpu().numpy(), mu, atol=1e-5)
    assert np.allclose(model.var_.cpu().numpy(), var, atol=1e-4)


def test_tensor_input_dtype_preserved():
    """Regression: a tensor input must be cast to the requested dtype.

    Pre-fix, _validate_data only applied `dtype` to numpy inputs; an existing
    float32 tensor stayed float32 even when dtype=torch.float64 was requested.
    """
    X = torch.tensor(data1[:500], dtype=torch.float32, device=DEVICE)
    model = IncrementalPCAonGPU(n_components=5)
    out = model.fit_transform(X, dtype=torch.float64)
    assert out.dtype == torch.float64


def test_weight_column_shape_and_length_validation():
    """Column-shaped weights must broadcast; mismatched lengths must raise."""
    X = _to_f64(data1[:200])
    w_row = torch.rand(X.shape[0], dtype=torch.float64, device=DEVICE)

    # A (n, 1) column vector should behave identically to a flat (n,) vector.
    model_flat = IncrementalPCAonGPU(n_components=3).fit(X, w_row)
    model_col = IncrementalPCAonGPU(n_components=3).fit(X, w_row.reshape(-1, 1))
    assert_subspaces_close(
        model_flat.components_.cpu().numpy(),
        model_col.components_.cpu().numpy(),
        atol=1e-6,
    )

    with pytest.raises(ValueError):
        IncrementalPCAonGPU(n_components=3).fit(X, torch.rand(X.shape[0] + 7, device=DEVICE))


if __name__ == "__main__":
    test_fit_method()
    test_partial_fit_method()
    test_incremental_mean_and_var_unweighted()
    test_incremental_mean_and_var_weighted()
    test_weighted_first_pass_matches_weighted_pca()
    test_weighted_partial_fit_method()
    test_weighted_fit_matches_closed_form()
    test_fit_transform_weights_default_and_dtype()
    test_noise_variance_wide_data()
    test_weighted_multi_batch_partial_fit_matches_closed_form()
    test_uniform_weights_match_unweighted()
    test_batch_size_invariance_weighted()
    test_weighted_mean_and_var_attributes()
    test_tensor_input_dtype_preserved()
    test_weight_column_shape_and_length_validation()
    test_incremental_mean_and_var_unweighted()
    test_incremental_mean_and_var_weighted()
    test_weighted_first_pass_matches_weighted_pca()
    test_weighted_partial_fit_method()
    test_weighted_fit_matches_closed_form()
    test_fit_transform_weights_default_and_dtype()
    test_noise_variance_wide_data()