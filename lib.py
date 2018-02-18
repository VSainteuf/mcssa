#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:08:37 2018

@author: viv
"""
import sys
import numpy as np
import pandas as pd

import utils
import ar1_fitting as ar1


class SSA:
    def __init__(self, data):
        self.data = np.array(data)  # store the data in the object
        try:
            self.index = list(data.index)
        except TypeError:
            self.index = [i for i in range(self.data.shape[0])]
        self.M = None  # Window length
        self.N2 = None  # Reduced length
        self.X = None  # Trajectory matrix
        self.covmat = None  # Covariance matrix
        self.E = None  # EOFs, matrix
        self.RC = None  # Reconstructed components
        self.values = None  # eigenvalues
        self.algo = None  # algorithm used for covariance matrix
        self.freqs = None  # frequencies of the EOFs
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
        self._embed(M)
        self._compute_cov(algo=algo)

        self.values, self.E = utils.eigen_decomp(self.covmat)
        self.freqs = utils.dominant_freqs(self.E)
        self.freq_rank = self.freqs.argsort()
        self.RC = utils.RC_table(self)

    def plot(self, freq_rank=True):
        return utils.plot(self,freq_rank)

    def show_f(self):
        return utils.freq_table(self)

    def reconstruct(self, components):
        if len(components) == 2:
            name = 'RC ' + str(components[0] + 1) + \
                '-' + str(components[1] + 1)
        else:
            name = 'Reconstruction'
        res = pd.DataFrame(index=self.index, columns=[name])
        res[name] = sum(self.RC.iloc[:, i] for i in range(len(components)))
        return res


class MCSSA(SSA):

    def __init__(self, data):
        super(MCSSA, self).__init__(data)
        # store the data in the object
        self.data = np.array(data) - np.mean(data)
        self.ar = AR()
        self.filtered_components = None
        self.stats = None
        self.scores = None
        self.ismc = True

    def run_mcssa(self, M, algo='BK', n_suro=100, filtered_components=[], level=5):
        # Store parameters
        self.filtered_components = filtered_components
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

            sys.stdout.write('\r Suroggate #' + str(i + 1) + '/' + str(n_suro))
            sys.stdout.flush()

        # Compute statistics of the surrogates projections and store them
        self.stats = utils.stats(samples, level)
        self.scores = utils.significance(samples, self.values)
        print('\n MCSSA completed!')

    def plot(self, freq_rank=True):
        return utils.plot(self,freq_rank)


class AR():
    def __init__(self):
        self.gamma = None
        self.alpha = None
        self.c0 = None
        self.N = None

    def set_parameters(self, mcssa):
        self.gamma, self.alpha, self.c0 = ar1.ar1comp(mcssa)
        self.N = mcssa.data.shape[0]

    def generate(self):
        suro = np.zeros(self.N)
        for i in range(1, self.N):
            suro[i] = self.gamma * suro[i - 1] + \
                self.alpha * np.random.normal()
        return suro


if __name__ == '__main__':
    # generate a test series:
    T = 8
    series = [np.sin(2 * np.pi / T * i) + np.random.rand() for i in range(100)]
    dfseries = pd.DataFrame(series)
    pdseries = pd.Series(series)
    ssa = SSA(series)
    ssa.run_ssa(20)
    ssa.plot(True)

    mcssa = MCSSA(series)
    mcssa.run_mcssa(20, n_suro=100, filtered_components=[0, 1, 2])
    mcssa.plot()

    mcssa2 = MCSSA(pdseries)
    mcssa2.run_mcssa(20, n_suro=100, filtered_components=[0, 1, 2])
    mcssa2.plot()
