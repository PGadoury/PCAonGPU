"""Optional benchmark; no large tensors or GPU allocations at import time."""

import argparse
import time

import torch
from sklearn.decomposition import IncrementalPCA

from gpu_pca import IncrementalPCAonGPU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    torch.manual_seed(0)
    X = torch.randn(args.samples, args.features)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def timed(label, fn):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"{label}: {time.perf_counter() - start:.3f}s")

    timed(
        "sklearn fit",
        lambda: IncrementalPCA(
            n_components=args.components, batch_size=args.batch_size
        ).fit(X.numpy()),
    )
    for backend in ["full", "gram", "lowrank"]:
        kwargs = {backend: True} if backend != "full" else {}
        model = IncrementalPCAonGPU(
            n_components=args.components,
            batch_size=args.batch_size,
            device=device,
            **kwargs,
        )
        model.fit(X[: max(args.components, args.batch_size)])  # Warm up kernels.
        timed(f"{device} {backend} fit (including transfer)", lambda: model.fit(X))


if __name__ == "__main__":
    main()
