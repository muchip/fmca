# FMCA
**Fast multiresolution covariance approximation**

Currently, the library features the construction of samplet bases and different versions of the pivoted Cholesky decomposition.
The samplet covariance compression introduced in 
[Samplets: Construction and scattered data compression](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4053305)
will be added soon.

Different scaling distributions and samplets on a Sigma shaped point cloud my look for example like depicted below.
![What is this](assets/samplets.png)

Representing an exponential covariance kernel with respect to this basis and truncating small entries leads to a sparse matrix
which can be factorized using nested dissection
![What is this](assets/compressed_kernel.png)
The left panel shows the kernel matrix, the middle panel the reordered matrix and the right panel the Cholesky factor.


## Installation
FMCA is a header only library. It depends on [Eigen](https://eigen.tuxfamily.org),
which has to be installed in advance.

Moreover, thanks to [pybind11](https://github.com/pybind/pybind11), FMCA may be compiled into a python module.
To this end, pybind11 needs to be installed as well. Afterwards, the module can simply be compiled using cmake:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make
```
example files and the compiled library are then located in build/py

## Samplets

FMCA features a samplet basis, which can be used to localize a given signal in the frequency domain. Given for example a
signal sampled at 100000 random locations like
![What is this](assets/signal.png)

the first 500 coefficients of the transformed signal looks like this
![What is this](assets/Tsignal.png)

The example above can be found and modified in the jupyter notebook FMCA_Samplets
