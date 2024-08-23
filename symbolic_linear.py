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

def substitute_to_array(expr_arr,x,xn):
    if len(expr_arr.shape) == 1:
        for i,expr in enumerate(expr_arr):
            expr = substitute_numbers(expr,x,xn)
            expr_arr[i] = expr
    else:
        if len(expr_arr.shape) == 2:
            for (i,j),expr in np.ndenumerate(expr_arr):
                expr = substitute_numbers(expr, x, xn)
                expr_arr[i][j] = expr

    return expr_arr

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

    def substitute_weight_and_biases(self,w,b):
          self.weight = nn.Parameter(torch.from_numpy(w).reshape(self.weight.shape))
          self.bias   = nn.Parameter(torch.from_numpy(b).reshape(self.bias.shape))

    def symbolicEvaluate(self,*args):
        if len(args) >= 1:
            xx = args[0]
            expr = np.matmul(xx,self.sym_weight.T)
            #self.sym_bias = array_of_vars('b',expr.shape[1],expr.shape[0])
            expr += self.sym_bias

        if len(args) == 2:
            xx_vals = args[1]


            #substitute xx values
            s  = expr.copy()
            s = substitute_to_array(s,xx,xx_vals)
            s = substitute_to_array(s, self.sym_weight,self.weight)
            s = substitute_to_array(s, self.sym_bias,self.bias)

            #
            #         for w,wn in zip(self.sym_weight,self.weight):
            #             s = s.subs(w[0], wn)
            #
            #         for w,wn in zip(self.sym_bias,self.bias):
            #             s = s.subs(w[0], wn)
            #
            #         sx[i] = s


        return expr,s



        fc1 = fc1.squeeze()

        return fc1

    def get_all_vars_and_values(self):
        res = [(self.sym_weight,self.weight.detach().numpy()),(self.sym_bias,self.bias.detach().numpy())]
        return res
    def get_all_vars(self):
        res = [self.sym_weight,self.sym_bias]
        return res

#var_list must be a list of tuples: (variable,value)
# expr_arr is an array of expressions
def substitute_all_vars(expr_arr_in,var_list):

    expr_arr = expr_arr_in.copy()
    for v in var_list:
        expr_arr = substitute_to_array(expr_arr,v[0],v[1])
    return expr_arr






if __name__ == '__main__':
    # make_symbolic_numpy_array_with_dimension_from_tensor('w',torch.ones(3))


    fc = nn.Linear(3,3)
    sfc = SymbolicLinear(3,3)
    x = torch.ones(3)
    xx = monkey_tensor('x', x)
    expr,s = sfc.symbolicEvaluate(xx, np.ones((3)))
    from sym_sigmoid import SymbolicSigmoid
    sig = SymbolicSigmoid(expr)

    from sym_diff import symbolic_diff
    d_sig = symbolic_diff(sig,w_00)

    #x = torch.ones((1,3))
    # y = fc(x)
    # y1 = sfc(x)


    #expr  = sin(x_00+x_10)
    # s = substitute_numbers(expr,xx,np.ones((2,2)))

#    s = sfc.symbolicEvaluate(xx,np.ones(1))



    qq = 0