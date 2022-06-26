# Triangle-based Outlier Detection

This repository contains the code used to evaluate the methods proposed in the paper [Triangle-based Outlier Detection](https://doi.org/10.1016/j.patrec.2022.03.008). 



## Usage example

Example dataset obtained from [Outlier Detection DataSets (ODDS)](http://odds.cs.stonybrook.edu/breast-cancer-wisconsin-original-dataset/).

```
from scipy.io import loadmat

import numpy as np
import pandas as pd

from TOD_code.methods import dis_matrices, TOD, sTOD

mat = loadmat('data/breastw.mat')
dataset = mat["X"]
labels = mat["y"]
labels = labels.reshape(labels.shape[0])

th = 0.65
k = range(245, 255)
z_size = 100
n_iter = 50

distmats = dis_matrices(dataset, k)
pred_TOD = TOD(distmats, th)
pred_sTOD = sTOD(distmats, th, z_size, n_iter)

```
