import torch
import numpy as np
from sklearn.decomposition import IncrementalPCA as SklearnIPCA
from gpu_pca.pca_module import IncrementalPCAonGPU

def generate_data(n_samples=50000, n_features=100, random_state=None):
    """Generate random Gaussian data."""
    rng = np.random.default_rng(random_state)
    return rng.standard_normal(size=(n_samples, n_features))

data1 = generate_data()
data2 = generate_data()

data1gpu = torch.tensor(data1, device='cuda')
data2gpu = torch.tensor(data2, device='cuda')

def test_fit_method():
    sklearn_model = SklearnIPCA(n_components=5)
    our_model = IncrementalPCAonGPU(n_components=5)

    sklearn_model.fit(data1)
    our_model.fit(data1gpu)

    transformed_sklearn = sklearn_model.transform(data1)
    transformed_our_model = our_model.transform(data1gpu).cpu().numpy()

    print(transformed_sklearn)
    print(transformed_our_model)
    
    # assert torch.allclose(torch.tensor(transformed_sklearn), torch.tensor(transformed_our_model), atol=1e-3)

def test_partial_fit_method():
    sklearn_model = SklearnIPCA(n_components=5)
    our_model = IncrementalPCAonGPU(n_components=5)

    sklearn_model.partial_fit(data1)
    sklearn_model.partial_fit(data2)

    our_model.partial_fit(data1gpu)
    our_model.partial_fit(data2gpu)

    transformed_sklearn = sklearn_model.transform(data1)
    transformed_our_model = our_model.transform(data1gpu).cpu().numpy()

    assert torch.allclose(torch.tensor(transformed_sklearn), torch.tensor(transformed_our_model), atol=5e-2)

def test_weighted_partial_fit_method():
    # Fit is somewhat sensitive to initial weights
    data1_weights = torch.rand(data1gpu.shape[0])**2 #0.5 * torch.ones(data1gpu.shape[0])
    data1_weights = torch.maximum(data1_weights, 1 - data1_weights)
    data1_weights_complement = 1 - data1_weights

    our_model = IncrementalPCAonGPU(n_components=5)
    our_model_weighted = IncrementalPCAonGPU(n_components=5)

    our_model.partial_fit(data1gpu)

    our_model_weighted.partial_fit(data1gpu, data1_weights)
    our_model_weighted.partial_fit(data1gpu, data1_weights_complement)

    #assert torch.allclose(torch.tensor(our_model.components_), torch.tensor(our_model_weighted.components_), atol=0.1)
    print(our_model.components_)
    print(our_model_weighted.components_)


if __name__ == "__main__":
    test_fit_method()
    test_partial_fit_method()
    test_weighted_partial_fit_method()