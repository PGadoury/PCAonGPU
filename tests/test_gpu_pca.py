"""Deterministic numerical references and public API regression tests."""

import numpy as np
import pytest
import torch
from sklearn.decomposition import IncrementalPCA as SklearnIPCA

from gpu_pca import IncrementalPCAonGPU as PCA

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(17)
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def data(n=61, d=5, dtype=torch.float64):
    return torch.randn(n, d, dtype=dtype) * torch.arange(1, d + 1) + 3


def covariance(model):
    return (
        model.components_.mH
        @ torch.diag(model.singular_values_.square()).to(model.components_.dtype)
        @ model.components_
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_size", [13, 100])
def test_weighted_covariance(device, dtype, batch_size):
    X = data(dtype=dtype)
    w = torch.rand(len(X), dtype=X.real.dtype) + 0.1
    w[::7] = 0
    model = PCA(n_components=5, batch_size=batch_size, device=device).fit(
        X, X_weights=w
    )
    mu = (X * w[:, None]).sum(0) / w.sum()
    centered = X - mu
    scatter = centered.mH @ (centered * w[:, None])
    dof = w.sum() - w.square().sum() / w.sum()
    tol = 2e-4 if X.element_size() <= (8 if X.is_complex() else 4) else 1e-9
    torch.testing.assert_close(
        model.mean_.cpu(), mu.to(model.mean_.dtype), rtol=tol, atol=tol
    )
    torch.testing.assert_close(covariance(model).cpu(), scatter, rtol=tol, atol=tol)
    torch.testing.assert_close(
        model.var_.cpu(),
        (scatter.diag().real / w.sum()).to(model.var_.dtype),
        rtol=tol,
        atol=tol,
    )
    torch.testing.assert_close(
        model.explained_variance_.cpu(),
        torch.linalg.eigvalsh(scatter).flip(0) / dof,
        rtol=tol,
        atol=tol,
    )
    torch.testing.assert_close(
        model.transform(X).cpu(),
        centered @ model.components_.cpu().mH,
        rtol=tol,
        atol=tol,
    )
    assert model.n_samples_seen_ == len(X)


@pytest.mark.parametrize("device", DEVICES)
def test_sklearn_incremental(device):
    X = data(90, 6)
    ref = SklearnIPCA(n_components=3, batch_size=15).fit(X.numpy())
    model = PCA(n_components=3, batch_size=15, device=device).fit(X)
    torch.testing.assert_close(
        covariance(model).cpu(),
        torch.from_numpy(
            ref.components_.T @ np.diag(ref.singular_values_**2) @ ref.components_
        ),
        rtol=1e-8,
        atol=1e-8,
    )
    for name in ["mean_", "var_", "explained_variance_", "explained_variance_ratio_"]:
        torch.testing.assert_close(
            getattr(model, name).cpu(),
            torch.as_tensor(getattr(ref, name)),
            rtol=1e-8,
            atol=1e-8,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["full", "gram", "lowrank"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_backends(device, backend, dtype):
    X = data(24, 40, dtype)
    opts = (
        {"gram": True}
        if backend == "gram"
        else {"lowrank": True, "lowrank_q": 24, "lowrank_seed": 3}
        if backend == "lowrank"
        else {}
    )
    model = PCA(n_components=4, device=device, **opts).fit(X)
    ref = PCA(n_components=4, device=device).fit(X)
    torch.testing.assert_close(covariance(model), covariance(ref), rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(
        model.noise_variance_, ref.noise_variance_, rtol=1e-8, atol=1e-8
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gram_degenerate(device, dtype):
    model = PCA(n_components=3, gram=True, device=device).fit(
        torch.zeros(6, 12, dtype=dtype)
    )
    c = model.components_
    torch.testing.assert_close(c @ c.mH, torch.eye(3, device=device, dtype=dtype))
    assert torch.isfinite(model.explained_variance_ratio_).all()
    assert not model.singular_values_.any()


@pytest.mark.parametrize("device", DEVICES)
def test_refit_rank_and_variance(device):
    model = PCA(device=device, batch_size=10).fit(data(21, 3))
    assert model.components_.shape == (3, 3)
    assert model.n_components is None
    model.fit(torch.tensor([[10.0], [12.0]]))
    assert model.n_samples_seen_ == 2
    assert model.mean_.item() == 11
    model = PCA(n_components=1, device=device)
    model.partial_fit(torch.tensor([[0.0], [2.0]]))
    model.partial_fit(torch.tensor([[4.0]]))
    assert model.var_.item() == pytest.approx(8 / 3)
    assert model.explained_variance_ratio_.item() == pytest.approx(1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_whiten_copy_and_dtype(device, dtype):
    X = data(60, 4, dtype).to(device)
    saved = X.clone()
    model = PCA(n_components=4, whiten=True, copy=False, device=device)
    out = model.fit_transform(X)
    torch.testing.assert_close(
        out.var(dim=0),
        torch.ones(4, device=device, dtype=X.real.dtype),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(out, model.transform(saved))
    model = PCA(n_components=2, device=device).fit(
        saved, dtype=torch.complex128 if dtype.is_complex else torch.float64
    )
    before = saved.clone()
    model.transform(saved)
    torch.testing.assert_close(saved, before)
    assert model.transform(saved).dtype == model.components_.dtype


@pytest.mark.parametrize("device", DEVICES)
def test_weights_normalization_and_shapes(device):
    X = data(40, 3)
    ref = PCA(n_components=3, batch_size=10, device=device).fit(X)
    for w in [2.0, torch.full((40,), 2.0), torch.full((40, 1), 2.0)]:
        model = PCA(n_components=3, batch_size=10, device=device).fit(X, X_weights=w)
        torch.testing.assert_close(model.explained_variance_, ref.explained_variance_)
        torch.testing.assert_close(model.mean_, ref.mean_)
    w = torch.rand(40, dtype=torch.float64)
    a = PCA(n_components=3, device=device).fit(X, X_weights=w)
    b = PCA(n_components=3, device=device).fit(X, X_weights=w / 1000)
    torch.testing.assert_close(a.explained_variance_, b.explained_variance_)
    before = a.components_.clone()
    a.partial_fit(X, X_weights=torch.zeros(40))
    torch.testing.assert_close(a.components_, before)
    assert a.n_samples_seen_ == 40
    c = PCA(n_components=1, device=device).fit([[1.0], [3.0]], X_weights=[0.25, 0.25])
    assert c.explained_variance_.item() == pytest.approx(2)


@pytest.mark.parametrize(
    "weights",
    [
        [-1.0, 1.0],
        [float("nan"), 1.0],
        [float("inf"), 1.0],
        [1.0, 2.0, 3.0],
        [[1.0, 2.0]],
        [1j, 2j],
        [0.0, 0.0],
    ],
)
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        PCA(device="cpu").fit([[1.0], [2.0]], X_weights=weights)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_components": 0},
        {"n_components": -1},
        {"n_components": True},
        {"batch_size": 0},
        {"lowrank": True, "gram": True},
        {"lowrank_q": 1, "n_components": 2},
        {"gram_eps": 0},
        {"stats_dtype": torch.float16},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        PCA(**kwargs)


@pytest.mark.parametrize(
    "X", [[], [[float("nan")]], [[float("inf")]], torch.empty(0, 3), torch.ones(3)]
)
def test_invalid_input(X):
    with pytest.raises(ValueError):
        PCA(device="cpu").fit(X)


def test_validation_and_positional_compatibility():
    X = data(20, 3)
    model = PCA(n_components=2, device="cpu")
    with pytest.raises(ValueError):
        model.transform(X)
    model.fit(X, False)  # Existing second positional argument remains check_input.
    with pytest.raises(ValueError):
        model.partial_fit(data(20, 4))
    with pytest.raises(ValueError):
        model.transform(X, dtype=torch.float32)
    with pytest.raises(ValueError):
        PCA(n_components=4).fit(X)
    with pytest.raises(ValueError):
        PCA(n_components=3).partial_fit(X[:2])
    with pytest.raises(ValueError):
        PCA().fit(X, dtype=torch.float16)
    with pytest.raises(ValueError):
        PCA().fit(X.to(torch.complex64), dtype=torch.float32)


@pytest.mark.parametrize("device", DEVICES)
def test_lowrank_residual_and_rng(device):
    X = data(30, 40).to(device)
    state = torch.random.get_rng_state().clone()
    model = PCA(
        n_components=3, lowrank=True, lowrank_q=3, lowrank_seed=2, device=device
    ).fit(X)
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    residual = (X - X.mean(0)).square().sum() - model.singular_values_.square().sum()
    torch.testing.assert_close(
        model.noise_variance_, residual / ((len(X) - 1) * (min(X.shape) - 3))
    )
    assert model.noise_variance_ > 0


@pytest.mark.parametrize("device", DEVICES)
def test_fit_streams_batches_and_preserves_input(device, monkeypatch):
    X = data(43, 4)
    saved = X.clone()
    model = PCA(n_components=3, batch_size=10, device=device)
    original = model._prepare
    seen = []

    def prepare(batch, *args, **kwargs):
        seen.append((batch.shape[0], batch.device.type))
        return original(batch, *args, **kwargs)

    monkeypatch.setattr(model, "_prepare", prepare)
    model.fit(X)
    assert seen == [(10, "cpu")] * 4 + [(3, "cpu")]
    torch.testing.assert_close(X, saved)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("backend", ["gram", "lowrank"])
def test_weighted_backends_multiple_batches(device, dtype, backend):
    X = data(60, 32, dtype)
    w = torch.rand(60, dtype=torch.float64) + 0.1
    kwargs = (
        {"gram": True}
        if backend == "gram"
        else {"lowrank": True, "lowrank_q": 20, "lowrank_seed": 0}
    )
    model = PCA(n_components=3, batch_size=16, device=device, **kwargs).fit(
        X, X_weights=w
    )
    ref = PCA(n_components=3, batch_size=16, device=device).fit(X, X_weights=w)
    torch.testing.assert_close(covariance(model), covariance(ref), rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(
        model.noise_variance_, ref.noise_variance_, rtol=1e-8, atol=1e-8
    )


@pytest.mark.parametrize("device", DEVICES)
def test_gram_failure_and_tall_fallback(device, monkeypatch):
    X = data(8, 16)
    ref = PCA(n_components=3, device=device).fit(X)

    def fail(*args, **kwargs):
        raise torch.linalg.LinAlgError("simulated failure")

    monkeypatch.setattr(torch.linalg, "eigh", fail)
    model = PCA(n_components=3, gram=True, batch_size=20, device=device).fit(X)
    torch.testing.assert_close(covariance(model), covariance(ref))
    X = data(20, 4)
    model = PCA(n_components=3, gram=True, batch_size=20, device=device).fit(X)
    ref = PCA(n_components=3, device=device).fit(X)
    torch.testing.assert_close(covariance(model), covariance(ref))


@pytest.mark.parametrize("scale", [1e-5, 1.0, 1e5])
def test_gram_scale(scale):
    X = data(10, 30) * scale
    model = PCA(n_components=3, gram=True, device="cpu").fit(X)
    ref = PCA(n_components=3, device="cpu").fit(X)
    torch.testing.assert_close(
        model.singular_values_, ref.singular_values_, rtol=1e-8, atol=1e-12
    )
    torch.testing.assert_close(
        model.noise_variance_, ref.noise_variance_, rtol=1e-8, atol=1e-12
    )


@pytest.mark.parametrize("device", DEVICES)
def test_single_effective_sample_and_zero_prefix(device):
    model = PCA(n_components=1, whiten=True, device=device).fit([[1.0]])
    assert model.explained_variance_.item() == 0
    assert model.transform([[1.0]]).item() == 0
    X = data(12, 3)
    model = PCA(n_components=3, batch_size=4, device=device).fit(
        X, X_weights=[0.0] * 4 + [1.0] * 8
    )
    ref = PCA(n_components=3, batch_size=4, device=device).fit(X[4:])
    torch.testing.assert_close(covariance(model), covariance(ref))
    assert model.n_samples_seen_ == 8


def test_precision_context_restored_on_failure(monkeypatch):
    old = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    model = PCA(
        n_components=2, device="cpu", matmul_precision="medium", allow_tf32=False
    )

    def fail(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(model, "_svd_fn_full", fail)
    with pytest.raises(RuntimeError):
        model.fit(data())
    assert torch.get_float32_matmul_precision() == old
    assert torch.backends.cuda.matmul.allow_tf32 == tf32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_rng_preserved():
    X = data().cuda()
    state = torch.cuda.get_rng_state().clone()
    model = PCA(n_components=2, device="cuda", lowrank=True, lowrank_seed=7)
    model.fit(X)
    torch.testing.assert_close(torch.cuda.get_rng_state(), state)
