# denmarf

Density EstimatioN using Masked AutoRegressive Flow

This package is basically a wrapper for [pytorch-flow](https://github.com/ikostrikov/pytorch-flows) that provides a sklearn-like interface to use pytorch to perform density estimation.

## Requirements
- scipy-stack (numpy, scipy, matplotlib, pandas)
- pytorch
- CUDA (for GPU capability)

## Installation
```
pip install .
```

## Usage
The interface is very similar to the [KernelDensity](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity) module in scikit-learn. To perform a density estimation, one first initialize a `DensityEstimate` object from `denmarf.density`. Then one fit the data, which is a numpy ndarray of size (n_samples, n_features)), `X` with the method `DensityEstimate.fit(X)`. Once a model is trained, it can be used to generate new samples using `DensityEstimate.sample()`, or to evaluate the density at arbitrary point with `DensityEstimate.score_samples()`

### Initializing a `DensityEstimate` object
To initialize a `DensityEstimate` model, one can simply use
```python
from denmarf.density import DensityEstimate

de = DensityEstimate()
```
Note that by default the model will try to use GPU whenever CUDA is available, and revert back to CPU if not available. To by-pass this behavior and use CPU even when GPU is available, use
```python
from denmarf.density import DensityEstimate

de = DensityEstimate(device="cpu", use_cuda=False)
```
If multiple GPUs are available, one can specify which device to use by
```python
from denmarf.density import DensityEstimate

de = DensityEstimate(device="cuda:2")
```
