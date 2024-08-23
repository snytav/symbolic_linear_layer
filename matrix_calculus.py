from sympy import *
import numpy as np


def diff_expression_array(expr_arr_in,x):
    expr_arr = expr_arr_in.copy()
    if len(expr_arr.shape) == 1:
        for i,expr in enumerate(expr_arr):
            expr_arr[i] = diff(expr,x)
        else:
            if len(expr_arr.shape) == 2:
                for (i,j),expr in np.ndenumerate(expr_arr):
                    expr_arr[i][j] = diff(expr, x)
    return expr_arr

def d_scalar_wrt_vector(s,vec):
    res = np.zeros(vec.shape[0],dtype=object)

    for i,v in enumerate(vec):
        res[i] = diff(s,v[0])
    return res

def d_vector_wrt_vector(vy,vx):
    res = np.zeros((vy.shape[0],vx.shape[0]),dtype=object)

    for i,y in enumerate(vy):
        res[i,:] = d_scalar_wrt_vector(y,vx)

    return res