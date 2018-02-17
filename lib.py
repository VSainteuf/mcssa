#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:08:37 2018

@author: viv
"""

import numpy as np
import pandas as pd

import utils


class ssa:
    def __init__(self,data):
            self.data = np.array(data) #store the data in the object
            self.M=None     #Window length
            self.N2=None    #Reduced length
            self.X=None     #Trajectory matrix
            self.covmat=None   #Covariance matrix
            self.E=None     #EOFs, matrix
            self.RC=None    #Reconstructed components
            self.values=None   #eigenvalues 
            self.algo=None     #algorithm used for covariance matrix
            self.freqs=None    #frequencies of the EOFs
            self.same_f=None
            self.ss_fft=None
            
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
    
    def _freq(self):
        self.freqs=utils.dominant_freqs(self.E)
        
    def RCs(self,M):
        if self.E is None:
            self.decomp(M)
        N2=self.data.shape[0]-self.M + 1
        x=self.data
        RC=pd.DataFrame(index=x.index,columns=['RC#'+str(i+1) for i in range(self.M)])
        
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
                RC.iloc[k,i]=np.diagonal(X2,offset=-(N2-1-k)).mean()    
        
        self.RC=RC
        
    
    def compute_ssa(self,M,algo='BK'):
        self._embed(M)
        self._compute_cov(algo=algo)
        self._decomp()
        self._freq()
#        self.RCs(M)
    
    def plot_spec(self,freq_rank=False):
        if not self.is_ready:
            raise ValueError('SSA has to be computed before plotting!')
        else:         
            if not freq_rank:
                fig=plt.figure()
                plt.yscale('log')
                plt.xlabel('Eigenvalue Rank')
                plt.ylabel('Variance')
                plt.title('M='+str(self.M)+'   (cov:' +str(self.algo)+')')
                plt.plot(self.values,marker='s',linewidth=0,color='r')
            
            else:
                freq=copy.copy(self.freqs)
                val=copy.copy(self.values)
                freq=np.array(freq)
                idx=freq.argsort()
                freq=freq[idx]
                val=val[idx]    
                fig=plt.figure()
                plt.yscale('log')
                plt.xlabel('Frequency (Cycle/t. unit)')
                plt.ylabel('Variance')
                plt.title('M='+str(self.M)+'   (cov:' +str(self.algo)+')')
                plt.plot(freq,val,marker='s',linewidth=0,color='r')
        
        return fig

    def show_f(self):
        tab=pd.DataFrame(index=range(self.M),columns=['EOF','Frequency'])
        freq=copy.copy(self.freqs)
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
        rec=pd.DataFrame(index=self.data.index,columns=[name])
        rec[name]=sum(self.RC['RC#'+str(components[i]+1)]  for i in range(len(components)))
        return rec

    
    def samef_test(self,level=0.75):
        if not self.is_ready :
            raise ValueError('SSA has not been computed yet, just do it')
        
        else:
            res=[]
            f=copy.copy(self.freqs)
            f=np.array(f)
            idx=f.argsort()
            f=f[idx]
            limit=level/(2*self.M)
            for i in range(self.M-1):
                delta=abs(f[i+1]-f[i])
                if delta<limit:
                    res.append((idx[i],idx[i+1]))
            for i in range(len(res)):
                print('EOFs# '+str(res[i][0]+1)+' & '+str(res[i][1]+1)+'  pass same fft test')
            self.same_f=res
            if len(res)==0:
                print('No pair found in same fft test, sorry')
                
    def strong_fft(self,level=2/3):
        if self.same_f is None:
             self.samef_test()
        candidates=copy.copy(self.same_f)
        res=[]
        for i in range(len(candidates)):
            nfft=2**11
            fft1=np.fft.fft(self.E[:,candidates[i][0]],axis=0,n=nfft)
            fft2=np.fft.fft(self.E[:,candidates[i][1]],axis=0,n=nfft)
            fft1=fft1[0:nfft//2]
            fft2=fft2[0:nfft//2]
            fft=(abs(fft1)**2+abs(fft2)**2)/self.M
            if fft.max()>level:
                res.append(candidates[i])
        for i in range(len(res)):
            print('EOFs# '+str(res[i][0]+1)+' & '+str(res[i][1]+1)+'  pass same AND strong fft test at '+str(self.freqs[res[i][0]])[0:5]+' cpu')
        self.ss_fft=res
        if len(res)==0:
            print('No pair found in strong fft test, sorry')