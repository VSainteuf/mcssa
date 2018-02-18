#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:14:00 2018

@author: viv
"""

import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import percentileofscore


#######################################################################
#Auxiliary functions for the SSA computations
#######################################################################

def embedded(x,M):
    """

    Args:
        x: numpy array, data time series
        shape: int tuple, shape of the desired embedding

    Returns:
        the embedded trajectory matrix
    """
    N2=x.shape[0] - M + 1
    xc=np.copy(x)
    X=np.zeros((N2,M))
    for i in range(N2):
        for j in range(M):
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

#######################################################################
#Auxiliary functions for the MCSSA computations
#######################################################################
    
def projection(series,E,algo='BK'):
    M=E.shape[0]
    N2=series.shape[0]-M+1
    Et=E.transpose()
    
    #Compute the series covariance matrix estimate
    if algo == 'BK':
        cseries = series - np.mean(series)
        Xr = embedded(cseries,M)
        Cr = covmat_bk(Xr,N2)
    elif algo == 'VG':
        Cr = covmat_vg(series,M)
    else:
        raise ValueError('Incorrect algorithm name')
    
    #Project on matrix E
    L = Et * Cr * E
    
    #Return diagonal elements of the projection
    return np.diagonal(L)


def stats(samples,level):
    M=samples.shape[1]
    
    descr=pd.DataFrame(index=['mean','2.5th perc.','97.5th perc.','rel_error inf','rel_error sup'],columns=['/EOF#'+str(i) for i in range(M)])

    descr.iloc[0,:]=np.mean(samples,axis=0)
    descr.iloc[1,:]=np.percentile(samples,level/2,axis=0)
    descr.iloc[2,:]=np.percentile(samples,100-level/2,axis=0)
    
    descr.iloc[3,:]=np.abs(descr.iloc[0,:] - descr.iloc[1,:])
    descr.iloc[4,:]=np.abs(descr.iloc[0,:]-descr.iloc[2,:])

    return descr

        
def significance(samples,values):
    scores=[]        
    for i in range(samples.shape[1]):
        scores+= [percentileofscore(samples[:,i],values[i],kind='weak')]
    return scores