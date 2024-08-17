import torch
import numpy as np
import torch.nn as nn
from sympy import *


def substitute_numbers(expr,x,xn):
    if len(x.shape) == 2:
        x = x.reshape(x.shape[0]*x.shape[1])
        xn = xn.reshape(xn.shape[0]*xn.shape[1])
    s = expr
    for i,x in enumerate(x):
        s = s.subs(x,xn[i])

    return s

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

def monkey_tensor(name,torch_tensor):
   npt = np.zeros_like(torch_tensor.detach().numpy())
   npt = npt.astype(object)

   if len(torch_tensor.shape) == 2:
      for (i,j),n in np.ndenumerate(npt):
           xi =var(name+'_%d%d' % (i,j))
           npt[i][j] = xi

      return npt

   if len(torch_tensor.shape) == 1:
      for i,n in enumerate(npt):
           xi =var(name+'_%d' % (i))
           npt[i] = xi

      return npt

class SymbolicLinear(nn.Linear):
    def __init__(self,n_in,n_out):
        super(SymbolicLinear,self).__init__(n_in,n_out)
        sym_weight = monkey_tensor('w',self.weight)
        sym_bias   = monkey_tensor('b',self.bias)
        self.sym_weight = sym_weight
        self.sym_bias   = sym_bias

    def symbolicEvaluate(self,*args):
        if len(args) >= 1:
            xx = args[0]
            expr = np.matmul(xx,self.sym_weight.T)
            #self.sym_bias = array_of_vars('b',expr.shape[1],expr.shape[0])
            expr += self.sym_bias

        if len(args) == 2:
            xx_vals = args[1]


            #substitute xx values
            for sx in expr:
                for i,s in enumerate(sx):
                    for x,v in zip(xx,xx_vals):
                        s = s.subs(x[0],v)

                    for w,wn in zip(self.sym_weight,self.weight):
                        s = s.subs(w[0], wn)

                    for w,wn in zip(self.sym_bias,self.bias):
                        s = s.subs(w[0], wn)

                    sx[i] = s


        return expr



        fc1 = fc1.squeeze()

        return fc1







if __name__ == '__main__':
    # make_symbolic_numpy_array_with_dimension_from_tensor('w',torch.ones(3))


    fc = nn.Linear(1,3)
    sfc = SymbolicLinear(1,3)

    x = torch.ones((1,2))
    # y = fc(x)
    # y1 = sfc(x)

    xx = monkey_tensor('x', x)
    expr  = sin(x_00+x_01)
    s = substitute_numbers(expr,xx,np.ones((2,2)))

#    s = sfc.symbolicEvaluate(xx,np.ones(1))



    qq = 0