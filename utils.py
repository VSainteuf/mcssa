#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:14:00 2018

@author: viv
"""


import numpy as np
import scipy as sp


#######################################################################
#Auxiliary functions for the SSA computations
#######################################################################

def embedded(x,shape):
    """

    Args:
        x: numpy array, data time series
        shape: int tuple, shape of the desired embedding

    Returns:
        the embedded matrix
    """
    M=shape[0]
    N2=shape[1]
    xc=np.copy(x)
    X=np.zeros(shape)
    for i in range(M):
        for j in range(N2):
            X[i][j]=xc[i+j]
    return np.matrix(X)

def covmat_bk(X,N2):
    """

    Args:
        X: numpy matrix, trajectory matrix
        N2: int, reduced length

    Returns:
        Covariance matrix estimator, following Broomhead&King method

    """
    return 1/N2 * X.transpose() * X

def covmat_vg(series,M):
    """

    Args:
        series: numpy array, data time series
        M: int, window length

    Returns:
        Covariance matrix estimator, following Vautard&Ghil method
    """
    N=series.shape[0]
    diag = np.zeros(M)

    for i in range(M):
        for k in range(N-i):
            diag[i]+=series[k]*series[k+i]
        diag[i]=diag[i]/(N-i)

    C=sp.linalg.toeplitz(diag)
    return np.matrix(C)

def eigen_decomp(matrix):
    """

    Args:
        matrix: numpy matrix

    Returns:
        values : list of eigenvalues in descending order
        E : eigenvectors matrix

    """

    values, E = sp.linalg.eig(matrix, right=True)
    idx = values.argsort()[::-1]
    values = values[idx]
    E = np.matrix(E[:, idx])

    return values, E

def dominant_freqs(E):
    """
    dominant frequencies of the column vectors of the input matrix
    Args:
        E: matrix of eigenvectors (as columns)

    Returns:
        list of dominant frequencies computed with fft
    """

    nfft = 2 ** 11
    fft = np.fft.fft(E, axis=0, n=nfft)
    fft = fft[0:nfft // 2]
    fft= np.abs(fft)**2
    freq = np.fft.fftfreq(nfft)
    freq = freq[0:nfft // 2]
    freqs = freq[fft.argmax(axis=0)]
    return freqs

