import torch
import numpy as np
import torch.nn as nn
from sympy import *

# Nx - horizontal
# By - vertical
def array_of_vars(name, Nx, Ny):
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

    def symbolicEvaluate(self,*args):
        if len(args) >= 1:
            xx = args[0]
            expr = np.matmul(xx,self.sym_weight.T)

        if len(args) == 2:
            xx_vals = args[1]


            #substitute xx values
            for sx in expr:
                for i,s in enumerate(sx):
                    for x,v in zip(xx,xx_vals):
                        s = s.subs(x[0],v)

                    for w,wn in zip(self.sym_weight,self.weight):
                        s = s.subs(w[0], wn)


                    sx[i] = s


            return



        fc1 = fc1.squeeze()

        return fc1







if __name__ == '__main__':
    fc = nn.Linear(1,3)
    sfc = SymbolicLinear(1,3)

    y = fc(torch.ones(1))
    y1 = sfc(torch.ones(1))

    xx = array_of_vars('x', 1, 1)

    s = sfc.symbolicEvaluate(xx,np.ones(1))



    qq = 0