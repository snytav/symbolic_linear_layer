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
        sym_weight = array_of_vars('w',n_in,n_out)
        sym_bias   = array_of_vars('b',n_out,n_out)
        self.sym_weight = sym_weight
        self.sym_bias   = sym_bias





if __name__ == '__main__':
    fc = nn.Linear(3,1)
    sfc = SymbolicLinear(3,1)

    y = fc(torch.ones(3))
    y1 = sfc(torch.ones(3))
    qq = 0