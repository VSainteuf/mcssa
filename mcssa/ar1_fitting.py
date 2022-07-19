#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:15:01 2018
AR(1) fitting algorithms (composite version)
For more details on the mathematical considerations, refer to:

Allen, Myles R., and Leonard A. Smith. "Monte Carlo SSA: Detecting Irregular
Oscillations in the Presence of Colored Noise."
Journal of Climate 9, no. 12 (1996): 3373-404.
http://www.jstor.org/stable/26201460.

@author: Vivien Sainte Fare Garnot
"""
import numpy as np
from scipy.optimize import newton
from scipy.linalg import toeplitz


def ar1comp(mcssa):
    """Determines the AR parameter for a given dataset,
        following Allen & Smith method
    Args:
        mcssa: MCSSA object for which the AR parameters need to be determined

    Returns:
        gamma,alpha,c0 : AR parameters

    """
    Cd = mcssa.covmat
    Ed = mcssa.E
    M = mcssa.M
    N = mcssa.data.shape[0]

    # Compute the noise projection matrix
    Q = proj_mat(Ed, mcssa.filtered_components)

    # Solve equation for gamma
    gam = solver(mcssa, Q)

    if gam > 1:
        gam = 0.99
        print('Non stationary value, setting gamma to 0.99')

    if gam < 0:
        gam = 0.01
        print('Negative value, setting gamma to 0.01')

    # Compute the other parameters
    c0 = trace0(Q * Cd * Q) / trace0(Q * Wp(gam, M, N) * Q)
    alpha = np.sqrt(c0 * (1 - gam**2))

    return gam, alpha, c0


def proj_mat(Ed, filtered_components):
    """Computes the projection matrix for a given set of filtered components
    Args:
        Ed (numpy matrix): EOF matrix
        filtered_components (list): the ranks of the EOFs to filter

    Returns:
        Q: numpy matrix
    """
    M = Ed.shape[0]
    if len(filtered_components) == 0:
        return 1
    d = np.ones(M)
    d[filtered_components] = 0
    K = np.diag(d)
    Q = Ed * K * Ed.transpose()

    return Q


def solver(mcssa, Q):
    """Solves the equation for the determination of gamma (see reference)
    Args:
        mcssa: MCSSA object
        Q (numpy matrix): projection matrix

    Returns:
        gam (float): the solution (only to this specific problem)

    """
    M = mcssa.M
    N = mcssa.data.shape[0]
    Cd = mcssa.covmat

    # Define the function to optimise and optimise
    q = trace1(Q * Cd * Q) / trace0(Q * Cd * Q)

    def fopt(gamma):
        return trace1(Q * Wp(gamma, M, N) * Q) / trace0(Q * Wp(gamma, M, N) * Q) - q

    try:
        gam = newton(fopt, q, tol=10e-5)
    except RuntimeError:
        gam = 0.99
        print('Algorithm failed to converge, setting gamma to 0.99')

    return gam


def Wp(g, M, N):
    c = g**np.arange(M) - musquare(g, N)
    w = toeplitz(c)
    w = np.matrix(w)
    return w


def musquare(gamma, N):
    """Bias correction function
    Args:
        gamma (float): the AR parameter
        N (int): length of the time series

    Returns:
        float value
    """
    res = float()
    res = -1 / N + 2 / N**2 * \
        ((N - gamma**N) / (1 - gamma) -
         (gamma * (1 - gamma**(N - 1))) / (1 - gamma)**2)
    return res


def trace0(m):
    """Averaged trace of the matrix
    Args:
        m (numpy matrix): input matrix

    Returns:
        float value
    """
    return 1 / m.shape[0] * float(m.trace())


def trace1(m):
    """Averaged offset trace of the matrix
    Args:
        m (numpy matrix): input matrix
    Returns:
        float value
    """
    return 1 / (m.shape[0] - 1) * float(m.trace(offset=1))
