#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:08:37 2018

@author: viv
"""

import numpy as np
import pandas as pd

import utils


class SSA:
    def __init__(self,data):
            self.data = np.array(data) #store the data in the object
            try:
                self.index = list(data.index)
            except TypeError:
                self.index = [i for i in range(self.data.shape[0])]
            self.M=None     #Window length
            self.N2=None    #Reduced length
            self.X=None     #Trajectory matrix
            self.covmat=None   #Covariance matrix
            self.E=None     #EOFs, matrix
            self.RC=None    #Reconstructed components
            self.values=None   #eigenvalues 
            self.algo=None     #algorithm used for covariance matrix
            self.freqs=None    #frequencies of the EOFs

            
    def _embed(self,M):
        self.M = M
        N=self.data.shape[0]
        self.N2=N-self.M+1

        if (self.N2<self.M):
            raise ValueError('Window length is too big')
        else:
            self.X = utils.embedded(self.data,(self.N2,self.M))

    def _compute_cov(self,algo='BK'):
        if algo=='BK':
            self.covmat = utils.covmat_bk(self.X,self.N2)

        elif algo=='VG':
            self.covmat = utils.covmat_vg(self.data,self.M)
        else:
            raise ValueError('Unknown covariance matrix algorithm name')
        
    def _decomp(self):
        self.values, self.E= utils.eigen_decomp(self.covmat)
    
    def _freqs(self):
        self.freqs=utils.dominant_freqs(self.E)
        
    def _RCs(self):
        x=self.data
        RC=pd.DataFrame(index=self.index,columns=['RC#'+str(i+1) for i in range(self.M)])
        
        #create the selection matrix

        for i in range(self.M):
            d=[0 for k in range(self.M)]
            d[i]=1
            I=np.diag(d)
        
            #Compute the filtered trajectory matrix
            X2=self.X*self.E*I*self.E.transpose()
            #switch antidiagonals to diagonals
            X2=np.flipud(X2)
            for k in range(x.shape[0]):
                RC.iloc[k,i]=np.diagonal(X2,offset=-(self.N2-1-k)).mean()
        
        self.RC=RC
        
    
    def compute_ssa(self,M,algo='BK'):
        self._embed(M)
        self._compute_cov(algo=algo)
        self._decomp()
        self._freqs()
        self._RCs()
    
    def plot(self,freq_rank=False):
        fig=plt.figure()
        plt.yscale('log')
        plt.ylabel('Variance')
        plt.title('M='+str(self.M)+'   (cov:' +str(self.algo)+')')


        if not freq_rank:
            plt.xlabel('Eigenvalue Rank')
            plt.plot(self.values,marker='s',linewidth=0,color='r')
        
        else:
            freq=np.copy(self.freqs)
            val=np.copy(self.values)
            idx=freq.argsort()
            freq=freq[idx]
            val=val[idx]    

            plt.xlabel('Frequency (Cycle/t. unit)')
            plt.plot(freq,val,marker='s',linewidth=0,color='r')
    
        return fig

    def show_f(self):
        tab=pd.DataFrame(index=range(self.M),columns=['EOF','f (in cycles per unit)'])
        freq=np.copy(self.freqs)
        freq=np.array(freq)
        idx=freq.argsort()
        freq=freq[idx]
        for i in range(self.M):
            tab.iloc[i,:]=['EOF '+str(idx[i]+1),freq[i]]
        print(tab)
        return tab
    
    def reconstruct(self,components):
        if len(components)==2:
            name='RC '+str(components[0]+1)+'-'+str(components[1]+1)
        else:
            name='Reconstruction'
        res=pd.DataFrame(index=self.index,columns=[name])
        res[name]=sum(self.RC.iloc[:,i+1]  for i in range(len(components)))
        return res
                

if __name__ == '__main__':
    #generate a test series:
    T=8
    series = [np.sin(2*np.pi/T * i) + np.random.rand() for i in range(100)]

    ssa = SSA(series)
    ssa.compute_ssa(20)
    ssa.plot(True)
    
    
    
    
    