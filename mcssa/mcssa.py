#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:08:37 2018

@author: Vivien Sainte Fare Garnot
"""
import sys
import numpy as np
import pandas as pd

import mcssa.utils as utils
import mcssa.ar1_fitting as ar1


class SSA:
    """Generic instance of a SSA analysis
    Args:
        data: array like, input time series, must be one dimensional

    Attributes:
        data (array): input time series
        index (index): index of the time series
        M (int): Window length
        N2 (int): Reduced length
        X (numpy matrix): Trajectory matrix
        covmat (numpy matrix): Covariance matrix
        E (numpy matrix): EOF matrix
        values (list): Eigenvalues
        RC (DataFrame): Reconstructed Components
        algo (str): Algorithm used to compute the covariance matrix
        freqs (list): Dominant frequencies of the EOFs
        freq_rank (index): Frequency ranked index
        ismc (bool): Monte Carlo test

    """
    def __init__(self, data):

        self.data = np.array(data)
        try:
            self.index = list(data.index)
        except TypeError:
            self.index = [i for i in range(self.data.shape[0])]
        self.M = None
        self.N2 = None
        self.X = None
        self.covmat = None
        self.E = None
        self.values = None
        self.RC = None
        self.algo = None
        self.freqs = None
        self.freq_rank = None
        self.ismc = False

    def _embed(self, M):
        self.M = M
        N = self.data.shape[0]
        self.N2 = N - self.M + 1

        if (self.N2 < self.M):
            raise ValueError('Window length is too big')
        else:
            self.X = utils.embedded(self.data, self.M)

    def _compute_cov(self, algo='BK'):
        if algo == 'BK':
            self.covmat = utils.covmat_bk(self.X, self.N2)

        elif algo == 'VG':
            self.covmat = utils.covmat_vg(self.data, self.M)
        else:
            raise ValueError('Incorrect algorithm name')
        self.algo = algo

    def run_ssa(self, M, algo='BK'):
        """Completes the Analysis on a SSA object

        Args:
            M (int): window length
            algo (str): covariance matrix algo ('BK' or 'VG')

        """
        self._embed(M)
        self._compute_cov(algo=algo)

        self.values, self.E = utils.eigen_decomp(self.covmat)
        self.freqs = utils.dominant_freqs(self.E)
        self.freq_rank = self.freqs.argsort()
        self.RC = utils.RC_table(self)

    def plot(self, freq_rank=True):
        """Plots the SSA spectrum of the series, assumes run_ssa has been completed
        Args:
            freq_rank (bool): if True eigen values are plotted against
            their dominant frequency, and if False in decreasing variance
            level

        Returns:
            matplotlib figure

        """
        return utils.plot(self, freq_rank)

    def show_f(self):
        """Shows the dominant frequency of each EOF
        Returns:
            pandas.DataFrame
        """
        return utils.freq_table(self)

    def reconstruct(self, components):
        """Computes the RC corresponding to a list of components
        Args:
            components (list): list of components

        Returns:
            pandas.Series

        """
        if len(components) == 2:
            name = 'RC {}-{}'.format(components[0] + 1, components[1] + 1)
        else:
            name = 'Reconstruction'

        components = [i-1 for i in components]
        res = self.RC.iloc[:, components].sum(axis=1)
        res.name = name
        res.index = self.index
        return res


class MCSSA(SSA):
    """Generic instance of an MC-SSA analysis.

    This class extends the SSA class to allow Monte Carlo testing of the
    SSA spectrum.
    The significance of the variance contained in each EOF is tested
    against the Null Hypothesis of an AR(1) process.
    If signal has already been identified in a given subset of EOFs,
    those can be filtered and not taken into acount in the significance
    test.

    Args:
        data (array): input time series, must be one dimensional

    Attributes:
        ar (mcssa.AR): AR instance attached to the MCSSA test
        filtered_components (list): list of filtered components
        stats (pandas.DataFrame): Descriptive statistics of the surrogate ensemble
        scores (list): Significance scores of the EOFs

    """
    def __init__(self, data):
        super(MCSSA, self).__init__(data)
        self.data = np.array(data) - np.mean(data)
        self.ar = AR()
        self.filtered_components = None
        self.stats = None
        self.scores = None
        self.ismc = True

    def run_mcssa(self, M,
                  algo='BK', n_suro=100, filtered_components=[], level=5):
        """Completes the MC-SSA algorithm on a MCSSA object instance

        Args:
            M (int) window lenght
            algo (str) covariance matrix algo ('BK' or 'VG')
            n_suro (int): number of surrogate realisations
            filtered_components (list): list of the filtered EOFs
            level (float): significance level in percent

        """
        # Store parameters,
        # EOFs are labelled from 0 to M-1 in the computations (not 1 to M)
        self.filtered_components = [i-1 for i in filtered_components]
        self.n_suro = n_suro

        # Compute SSA and determine AR1 parameters
        print('Computing parameters')
        self.run_ssa(M, algo=algo)
        self.ar.set_parameters(self)

        # Generate surrogate and store diagonal elements of the projection
        samples = np.zeros((n_suro, M))
        print('Generating surrogate ensemble')
        for i in range(n_suro):
            suro = self.ar.generate()
            samples[i, :] = utils.projection(suro, self.E, algo=self.algo)

            sys.stdout.write('\r Suroggate # {}/{}'.format(i + 1, n_suro))
            sys.stdout.flush()

        # Compute statistics of the surrogates projections and store them
        self.stats = utils.stats(samples, level)
        self.scores = utils.significance(samples, self.values)
        print('\n MCSSA completed!')

    def plot(self, freq_rank=True):
        """Plots the MCSSA spectrum, assumes run_mcssa has been completed

        Args:
            freq_rank (bool) if True eigenvalues are plotted
            in increasing frequency, and if False in decreasing variance
            level.

        Returns:
            matplotlib figure

        """
        return utils.plot(self, freq_rank)


class AR():
    """
    Class representing a generic AR(1) process defined by the formula:
    z(t+1) = gamma * z(t) + alpha * e(t) with e(t) a random normal process
    """

    def __init__(self):
        self.gamma = None  # parameter
        self.alpha = None  # parameter
        self.c0 = None  # lag-0 covariance
        self.N = None  # desired length for the realisations

    def set_parameters(self, mcssa):
        """
        Determines the AR parameters that will constitute
        the strongest null hypothesis
        Args:
            mcssa: MCSSA instance

        """
        self.gamma, self.alpha, self.c0 = ar1.ar1comp(mcssa)
        self.N = mcssa.data.shape[0]

    def generate(self):
        """
        Returns: a realisation of the AR process,
        assumes that parameters are set
        """
        suro = np.zeros(self.N)
        for i in range(1, self.N):
            suro[i] = self.gamma * suro[i - 1] + \
                self.alpha * np.random.normal()
        return suro


if __name__ == '__main__':
    # generate a test series with random noise and oscillatory signal:
    T = 8
    series = [np.sin(2 * np.pi / T * i) + np.random.rand() for i in range(100)]

    # SSA analysis
    ssa = SSA(series)
    ssa.run_ssa(20)
    ssa.plot()

    #Reconstruction
    RC23 = ssa.reconstruct([2,3])
    RC23.plot()

    # MCSSA analysis
    mcssa = MCSSA(series)
    mcssa.run_mcssa(20, n_suro=1000, filtered_components=[0, 1, 2])
    mcssa.plot()
    mcssa.show_f()


