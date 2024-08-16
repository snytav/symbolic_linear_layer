from sympy import *
import numpy as np
from symbolic_linear import SymbolicLinear,array_of_vars

class SymbolicSigmoid(SymbolicLinear):
    def symbolicEvaluate(self, *args):
        fc1 = args[0]
        sigma_fc1 = np.zeros_like(fc1)
        # sigmoid of 1st layer
        squarer = lambda t: 1 / (1 + exp(-t))
        if len(fc1.shape) == 1:
            sigma_fc1 = np.array([squarer(xi) for xi in fc1])
        else:
            if len(fc1.shape) == 2:

                sigma_fc1 = sigma_fc1.astype(object)
                for (i, j), xi in np.ndenumerate(xx):
                    sigma_fc1[i][j] = squarer(xi)

        return sigma_fc1


if __name__ == '__main__':
    xx = array_of_vars('x', 3, 1)

    sig = SymbolicSigmoid(3,1)

    s = sig.symbolicEvaluate(xx,np.ones((3,1)))

