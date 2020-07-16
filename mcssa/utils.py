#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:14:00 2018
Auxiliary functions for the computation and display of SSA and MC-SSA analyses

For more details on the mathematical considerations, refer to:
Allen, Myles R., and Leonard A. Smith. "Monte Carlo SSA: Detecting Irregular
Oscillations in the Presence of Colored Noise."
Journal of Climate 9, no. 12 (1996): 3373-404.
http://www.jstor.org/stable/26201460.
@author: Vivien Sainte Fare Garnot
"""

import numpy as np
from scipy.stats import percentileofscore
from scipy.linalg import toeplitz, eig
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# Auxiliary functions for the SSA computations
###############################################################################


def embedded(x, M):
    """Computes the M embedding of a time series

    Args:
        x (numpy array): data time series
        M (int): dimension of the desired embedding (window length)

    Returns:
        the embedded trajectory matrix
    """
    N2 = x.shape[0] - M + 1
    xc = np.copy(x)
    X = np.zeros((N2, M))
    for i in range(N2):
        for j in range(M):
            X[i][j] = xc[i + j]
    return np.matrix(X)


def covmat_bk(X, N2):
    """Covariance matrix estimator, following Broomhead&King method

    Args:
        X (numpy matrix): trajectory matrix
        N2 (int): reduced length

    Returns:
        numpy matrix

    """
    return 1 / N2 * X.transpose() * X


def covmat_vg(series, M):
    """Covariance matrix estimator, following Vautard & Ghil method

    Args:
        series (numpy array): data time series
        M (int): window length

    Returns:
        numpy matrix
    """
    N = series.shape[0]
    diag = np.zeros(M)

    for i in range(M):
        for k in range(N - i):
            diag[i] += series[k] * series[k + i]
        diag[i] = diag[i] / (N - i)

    C = toeplitz(diag)
    return np.matrix(C)


def eigen_decomp(matrix):
    """Performs the eigendecomposition of the input matrix.

    Args:
        matrix (numpy matrix): matrix to decompose

    Returns:
        values (list): eigenvalues arranged in descending order
        E (numpy matrix): eigenvectors matrix

    """

    values, E = eig(matrix, right=True)
    idx = values.argsort()[::-1]
    values = values[idx]
    E = np.matrix(E[:, idx])

    return values, E


def RC_table(ssa):
    """Computes the reconstructions of all M components and stores
    them in the M columns of a DataFrame.

    Args:
        ssa (mcssa.SSA): instance of SSA for which the reconstruction
        is computed

    Returns:
        pandas.Dataframe

    """
    RC = pd.DataFrame(index=ssa.index, columns=[
                      'RC#' + str(i + 1) for i in range(ssa.M)])

    for i in range(ssa.M):
        d = [0 for k in range(ssa.M)]
        d[i] = 1
        I = np.diag(d)
        # Compute the filtered trajectory matrix
        X2 = ssa.X * ssa.E * I * ssa.E.transpose()
        # switch antidiagonals to diagonals
        X2 = np.flipud(X2)

        for k in range(ssa.data.shape[0]):
            RC.iloc[k, i] = np.diagonal(X2, offset=-(ssa.N2 - 1 - k)).mean()

    return RC


def dominant_freqs(E):
    """Computes the dominant frequencies of the column vectors
    of the input matrix using FFT.

    Args:
        E (numpy matrix): Matrix of eigenvectors (as columns)

    Returns:
        list of floats
    """

    nfft = 2 ** 11
    fft = np.fft.fft(E, axis=0, n=nfft)
    fft = fft[0:nfft // 2]
    fft = np.abs(fft)**2
    freq = np.fft.fftfreq(nfft)
    freq = freq[0:nfft // 2]
    freqs = freq[fft.argmax(axis=0)]
    return freqs

###############################################################################
# Auxiliary functions for the MCSSA computations
###############################################################################


def projection(series, E, algo='BK'):
    """Computes the covariance matrix of the series and projects it onto E.
    Returns the diagonal elements of the projection

    Args:
        series (array): input time series, must be 1-dimensional
        E (numpy matrix): EOF matrix
        algo (str): covariance matrix algorithm ('BK' or 'VG')


    Returns:
        list of floats

    """
    M = E.shape[0]
    N2 = series.shape[0] - M + 1
    Et = E.transpose()

    # Compute the series covariance matrix estimate
    if algo == 'BK':
        cseries = series - np.mean(series)
        Xr = embedded(cseries, M)
        Cr = covmat_bk(Xr, N2)
    elif algo == 'VG':
        Cr = covmat_vg(series, M)
    else:
        raise ValueError('Incorrect algorithm name')

    # Project on matrix E
    L = Et * Cr * E

    # Return diagonal elements of the projection
    return np.diagonal(L)


def stats(samples, level):
    """Computes the descriptive statistics of the surrogate ensemble

    Args:
        samples (numpy array): pojections of all the surrogate realisations
        level (float): significance level in percent

    Returns:
        pandas.DataFrame

    """
    M = samples.shape[1]

    descr = pd.DataFrame(index=['mean', '2.5th perc.', '97.5th perc.',
                                'rel_error inf', 'rel_error sup'],
                         columns=['/EOF#' + str(i) for i in range(M)])

    descr.iloc[0, :] = np.mean(samples, axis=0)
    descr.iloc[1, :] = np.percentile(samples, level / 2, axis=0)
    descr.iloc[2, :] = np.percentile(samples, 100 - level / 2, axis=0)
    descr.iloc[3, :] = np.abs(descr.iloc[0, :] - descr.iloc[1, :])
    descr.iloc[4, :] = np.abs(descr.iloc[0, :] - descr.iloc[2, :])

    return descr


def significance(samples, values):
    """Computes the significance of the EOFs

    Args:
        samples (numpy array): pojections of all the surrogate realisations
        values (list): eigenvalues of the SSA analysis

    Returns:
        list of the significance of each eigenvalue as percentages

    """
    return [percentileofscore(samples[:, i], values[i], kind='weak')
            for i in range(samples.shape[1])]


###############################################################################
# Dsiplaying and plotting functions
###############################################################################


def plot(mc_ssa, freq_rank=True):
    """Plotting method for both SSA and MCSSA objects

    Args:
        mc_ssa: SSA or MCSSA object
        freq_rank (boolean): if true EOFs are plotted against their
        dominant frequency

    Returns:
        matplolib figure

    """

    fig = plt.figure()
    plt.yscale('log')
    plt.ylabel('Variance')
    plt.title('M: {}; cov: {}'.format(mc_ssa.M, mc_ssa.algo))

    if not freq_rank:
        x = [i for i in range(mc_ssa.M)]
        y = mc_ssa.values

        plt.xlabel('Eigenvalue Rank')
        plt.plot(y, marker='s', linewidth=0, color='r')

    else:
        x = mc_ssa.freqs[mc_ssa.freq_rank]
        y = mc_ssa.values[mc_ssa.freq_rank]

        plt.xlabel('Frequency (Cycle/t. unit)')
        plt.plot(x, y, marker='s', linewidth=0, color='r')

    if mc_ssa.ismc:
        errors = np.array(mc_ssa.stats.iloc[3:5, :])
        mean_suro = np.array(mc_ssa.stats.iloc[0, :])
        plt.errorbar(x, y=mean_suro, yerr=errors,fmt='none',
                     ecolor='k', elinewidth=.5, capsize=2.5)
        plt.title('M: {}; g: {}; a: {}; Ns: {}'.format(mc_ssa.M,
                                                       mc_ssa.ar.gamma,
                                                       mc_ssa.ar.alpha,
                                                       mc_ssa.n_suro))

    plt.show()

    return fig


def freq_table(mc_ssa):
    """Builds a table with the frequency of each EOF
    Args:
        mc_ssa: MCSSA object

    Returns:
        pandas.Dataframe

    """
    tab = pd.DataFrame(index=range(mc_ssa.M),
                       columns=['EOF', 'f (in cycles per unit)'])
    freq = mc_ssa.freqs[mc_ssa.freq_rank]
    for i in range(mc_ssa.M):
        tab.iloc[i, :] = ['EOF ' + str(mc_ssa.freq_rank[i] + 1), freq[i]]
    print(tab)
    return tab
