# FMCA

[![CI](https://github.com/muchip/fmca/actions/workflows/ci.yml/badge.svg)](https://github.com/muchip/fmca/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fmca.svg)](https://pypi.org/project/fmca/)

**Fast multiresolution covariance analysis**

FMCA is a header only library for the multiresolution analysis of scattered data and kernel matrices. It is developed
at the [Università della Svizzera italiana](https://www.usi.ch) in the research group of [Michael Multerer](http://usi.to/3ps).

Currently, the library features the construction of samplet bases and different versions of the pivoted Cholesky decomposition,
as well as the fast samplet covariance compression introduced in 
[Samplets: Construction and scattered data compression](https://doi.org/10.1016/j.jcp.2022.111616).

Different scaling distributions and samplets on a Sigma shaped point cloud may look for example like depicted below.
![What is this](https://raw.githubusercontent.com/muchip/fmca/master/assets/samplets.png)

Representing an exponential covariance kernel with respect to this basis and truncating small entries leads to a sparse matrix
which can be factorized using nested dissection
![What is this](https://raw.githubusercontent.com/muchip/fmca/master/assets/compressed_kernel.png)
The left panel shows the kernel matrix, the middle panel the reordered matrix and the right panel the Cholesky factor.


## Installation

FMCA is header only and depends on [Eigen](https://eigen.tuxfamily.org).
If Eigen is not installed, the build downloads it automatically. OpenMP is used
whenever the compiler supports it.

### Python module via pip

Thanks to [pybind11](https://github.com/pybind/pybind11), FMCA may be compiled
into a Python module. The easiest way is to install it directly from GitHub
(requires CMake ≥ 3.21, a C++17 compiler and Python ≥ 3.9):

    python3 -m pip install git+https://github.com/muchip/fmca@master

To select a compiler, e.g. an OpenMP-capable GCC on macOS where Apple clang
lacks OpenMP, set `CXX` before installing:

    CXX=g++-15 python3 -m pip install git+https://github.com/muchip/fmca@master

Afterwards `import FMCA` works in that Python environment. Update to the latest
commit with

    python3 -m pip install --upgrade --no-cache-dir git+https://github.com/muchip/fmca@master

### Building with CMake

For development, or to build the C++ tests, pybind11 needs to be installed
(`python3 -m pip install pybind11`). Then

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ../
    make

The compiled module and the example notebooks are located in `build/py`, the
tests in `build/tests`. Add `-DCMAKE_CXX_COMPILER=...` to the cmake call to
select a compiler; `-DFMCA_BUILD_TESTS=OFF` or `-DFMCA_BUILD_PYTHON=OFF` skip
the respective parts.

## Python interface

Points are passed as a `dim x N` array, one point per column, and data as an `N x k` array, one
row per point. The samplet transform works on data in the order of the cluster tree, which
`toClusterOrder` and `toNaturalOrder` take care of:
```python
import numpy as np
import FMCA

pts = np.random.rand(2, 10000)          # 10000 points in 2D, one per column
ST = FMCA.SampletTree(pts, 3)           # samplet tree with 3 vanishing moments
f = np.sin(4 * pts[0]).reshape(-1, 1)   # data, one row per point

c = ST.sampletTransform(ST.toClusterOrder(f))           # samplet coefficients
g = ST.toNaturalOrder(ST.inverseSampletTransform(c))    # back to the data
```

## Samplets

FMCA features a samplet basis, which can be used to localize a given signal in the frequency domain. Given for example a
signal sampled at 100000 random locations, e.g.,
![What is this](https://raw.githubusercontent.com/muchip/fmca/master/assets/signal.png)

the first 500 coefficients of the transformed signal looks like this
![What is this](https://raw.githubusercontent.com/muchip/fmca/master/assets/Tsignal.png)

The example above can be found and modified in the jupyter notebook FMCA_Samplets

## Denoising

Since samplets have vanishing moments, a smooth signal is represented by very few large
coefficients, whereas white noise is spread evenly over all of them. Discarding the small
coefficients thus removes most of the noise and almost none of the signal. 

For an image, the samplet transform is applied to its columns and then to its rows, and every
band of coefficients is thresholded on its own (BayesShrink), with the noise level estimated
from the data. For the 600x512 image below, perturbed by Gaussian noise of standard deviation
0.1, the result is on par with standard wavelet denoising.

![What is this](https://raw.githubusercontent.com/muchip/fmca/master/assets/denoising_image.png)

The denoising can be found and modified in the jupyter notebooks
[FMCA_SampletDenoising1D](https://github.com/muchip/fmca/blob/master/py/FMCA_SampletDenoising1D.ipynb)
and [FMCA_SampletImageDenoising](https://github.com/muchip/fmca/blob/master/py/FMCA_SampletImageDenoising.ipynb).

## Adaptive tree search

The samplet coefficients moreover indicate *where* a function fails to be smooth, which can be fed
back into the cluster tree. Refining the tree uniformly splits every cluster, no matter whether the
data require it or not.

![What is this](https://raw.githubusercontent.com/muchip/fmca/master/assets/uniform_clusters.png)

The Binev-DeVore algorithm instead activates the clusters carrying the largest amount of
energy, until the energy left outside the tree drops below a prescribed tolerance. The leaves of the
resulting adaptive tree form a partition of the point cloud which is fine only where the function
varies. 

In the example below, 100000 scattered points are partitioned into 128 clusters, while
resolving the whole domain at the same finest scale would require 2048 uniform ones.

![What is this](https://raw.githubusercontent.com/muchip/fmca/master/assets/adaptive_clusters.png)

The tolerance is measured relative to the energy of the data and is therefore scale free. The
example above can be found and modified in the jupyter notebook
[FMCA_SampletAdaptiveClustering](https://github.com/muchip/fmca/blob/master/py/FMCA_SampletAdaptiveClustering.ipynb).

## Gaussian process learning

FMCA provides different variants of the pivoted (truncated) Cholesky decomposition, cp.
[On the low-rank approximation by the pivoted Cholesky decomposition](https://www.sciencedirect.com/science/article/pii/S0168927411001814)
and the references therein, that can be used for Gaussian process learning.

posterior mean (red) and posterior standard deviation (green) conditioned on the blue dots
![What is this](https://raw.githubusercontent.com/muchip/fmca/master/assets/gaussian_process.png)

The example above can be found and modified in the jupyter notebook [FMCA_GP](https://github.com/muchip/fmca/blob/master/py/FMCA_GP.ipynb).

## Samplet Gaussian process filtering

A samplet matrix compression based approach is also available. It particular allows for filtering of the (compressed) kernel
matrix, thus mitigating the very ill-conditioning of the kernel matrix. 

![What is this](https://raw.githubusercontent.com/muchip/fmca/master/assets/kernel.png)![What is this](https://raw.githubusercontent.com/muchip/fmca/master/assets/filteredKernel.png)

For the Matern-3/2 kernel shown on the left, just considering the diagonal block associated to the 40 largest entries, shown on the right,
leads to a relative approximation error of about 3e-5 of the kernel matrix in the Frobenius norm. Solving the associated system for the noisy
data set shown below, leads to an effective denoising. The corresponding expectation is shown in orange.

![What is this](https://raw.githubusercontent.com/muchip/fmca/master/assets/filteredGP.png)

This example
can be found [FMCA_Samplet_GP_Filtering](https://github.com/muchip/fmca/blob/master/py/FMCA_Samplet_GP_Filtering.ipynb).


## Python notebooks

The notebooks below live in `py/` and are copied next to the compiled module in `build/py`,
so they can be run directly from there. They cover the samplet basis, the compression of
signals, images and kernel matrices, and the data driven refinement of the cluster tree.

| notebook | what it shows |
| --- | --- |
| [FMCA_Samplets](https://github.com/muchip/fmca/blob/master/py/FMCA_Samplets.ipynb) | samplet trees, what a samplet looks like in 1D and 2D, vanishing moments, orthogonality and sparsity of the transform, a signal in the samplet basis, and the compression of a kernel matrix |
| [FMCA_SampletCompression1D](https://github.com/muchip/fmca/blob/master/py/FMCA_SampletCompression1D.ipynb) | a signal in the natural basis versus the samplet basis, coefficient decay and best N-term approximation |
| [FMCA_SampletDenoising1D](https://github.com/muchip/fmca/blob/master/py/FMCA_SampletDenoising1D.ipynb) | hard and soft thresholding of samplet coefficients, and the universal threshold |
| [FMCA_SampletImageDenoising](https://github.com/muchip/fmca/blob/master/py/FMCA_SampletImageDenoising.ipynb) | compression and denoising of an image with the separable samplet transform, compared with standard wavelets |
| [FMCA_SampletKernelCompression](https://github.com/muchip/fmca/blob/master/py/FMCA_SampletKernelCompression.ipynb) | the dense kernel matrix versus its samplet compression, accuracy against cost, and the scaling in N |
| [FMCA_SampletAdaptiveClustering](https://github.com/muchip/fmca/blob/master/py/FMCA_SampletAdaptiveClustering.ipynb) | adaptive partitions of a point cloud obtained from the samplet coefficients |
| [FMCA_Samplet_GP_Filtering](https://github.com/muchip/fmca/blob/master/py/FMCA_Samplet_GP_Filtering.ipynb) | filtering of the compressed kernel matrix for Gaussian process regression |
| [FMCA_Samplet_KRR](https://github.com/muchip/fmca/blob/master/py/FMCA_Samplet_KRR.ipynb) | kernel ridge regression in the samplet basis |
| [FMCA_GP](https://github.com/muchip/fmca/blob/master/py/FMCA_GP.ipynb) | Gaussian process learning based on the pivoted Cholesky decomposition |
| [FMCA_Cholesky](https://github.com/muchip/fmca/blob/master/py/FMCA_Cholesky.ipynb) | the different variants of the pivoted Cholesky decomposition |
| [FMCA_H2Matrix](https://github.com/muchip/fmca/blob/master/py/FMCA_H2Matrix.ipynb) | H2-matrix construction and fast matrix-vector products |
| [FMCA_LowRankBenchmarks](https://github.com/muchip/fmca/blob/master/py/FMCA_LowRankBenchmarks.ipynb) | low-rank benchmarks for adaptive joint distribution learning |
