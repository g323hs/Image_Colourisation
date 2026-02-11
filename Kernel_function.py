#Gaussian phi
import math as m
import numpy as np

def phi_gauss(r):
    return m.exp(-r**2)

#Wendland RBF
def phi_Wendland(r):
    plus = max(0,1-r)
    return plus**4*(4*r+1)

def K_kernel(phi, gamma, x, y, sigma1, sigma2, p):
    """
    ---Inputs---
    phi [function]: the radial basis function (RBF)
    gamma [function]: grayscale information
    x [numpy array]: vector- the variable of the function
    y [numpy array]: vector - the 2nd variable of the function

    sigma1 [float]: parameter of the function
    sigma2 [float]: parameter of the function
    p [float]: parameter of the function

    ---Outputs---
    The kernel function value K(x,y) [numpy array]
    """
    norm_diff = np.linalg.norm(x-y,2)
    norm_gammadiff = np.linalg.norm(gamma(x)-gamma(y),2)
    K = phi(norm_diff/sigma1)*phi((norm_gammadiff**p)/sigma2)
    return K


