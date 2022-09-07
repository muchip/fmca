# %% [markdown]
# # FMCA interface

# %% [markdown]
# ### first import modules

# %%
# import seems necessary to not crash matplotlib
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import linalg as la
import time
import FMCA

# %%
dim = 2
N = 1000
cov = FMCA.CovarianceKernel("exponential", 2)
pts = np.array(np.random.randn(dim, N), order='F')

# %%
Chol = FMCA.PivotedCholesky(cov, pts, 1e-1)
L = Chol.matrixL()
Chol.computeFullPiv(cov, pts, 1e-1)
L2 = Chol.matrixL()

# %%
K = cov.eval(pts,pts)
Keps = np.matmul(L, L.transpose())
Keps2 = np.matmul(L2, L2.transpose())
print(np.linalg.norm(K - Keps) / np.linalg.norm(K))
print(np.linalg.norm(K - Keps2) / np.linalg.norm(K))

# %%



