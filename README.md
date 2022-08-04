# FMCA
**Fast multiresolution covariance approximation**

Currently, the library features the construction of samplet basis and different versions of the pivoted Cholesky decomposition.
The samplet covariance compression introduced in 
[Samplets: Construction and scattered data compression](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4053305)
will be added soon.

## Installation
FMCA is a header only library

The current implementation provides a pybind11
interface for python, which can be compiled using cmake:
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
