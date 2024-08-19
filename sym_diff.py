import numpy as np
from sympy import *

# derivative along single variable
def symbolic_diff(f, x):
    df = np.zeros_like(f)
    # sigmoid of 1st layer
    #squarer = lambda t: 1 / (1 + exp(-t))
    if len(f.shape) == 1:
        df_dx = np.array([diff(xi,x) for xi in f])
    else:
        if len(f.shape) == 2:
            df_dx = np.zeros_like(f)
            df_dx = df_dx.astype(object)
            for (i, j), xi in np.ndenumerate(f):
                df_dx[i][j] = diff(xi,x)

    return df_dx
