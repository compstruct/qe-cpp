# Pytorch C++ extension of Quantile Estimation

## Build
```
python setup.py install
```

## Usage
```
import torch
from qe_cpp import *

mask = qe(acc, q_init, q_step, q) # acc gradient, initial guess, quantile step size, target quantile
```

## Note
need gcc 5.0+
