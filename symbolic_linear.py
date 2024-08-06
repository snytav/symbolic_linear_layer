import torch
import numpy as np
import torch.nn as nn
from sympy import *

def array_of_vars(name, Ny, Nx):
    xx = []
    for i in range(Ny):
        xline = []
        for j in range(Nx):
            xl = var(name + '_%d%d' % (i, j))
            xline.append(xl)


        xline = np.array(xline)
        xline.squeeze()
        xx.append(xline)
    xx = np.array(xx)
    #xx = xx.reshape(xx.shape[0])
    return xx

class SymbolicLinear(nn.Linear):
    def __init__(self,n_in,n_out):
        super(SymbolicLinear,self).__init__(n_in,n_out)







if __name__ == '__main__':
    fc = nn.Linear(3,1)
    sfc = SymbolicLinear(3,1)

    qq = 0