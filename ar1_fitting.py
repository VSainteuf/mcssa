#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:15:01 2018
AR(1) fitting algorithms, standard and composite versions
For details on the maths and notations refer to :

Allen, Myles R., and Leonard A. Smith. "Monte Carlo SSA: Detecting Irregular 
Oscillations in the Presence of Colored Noise." 
Journal of Climate 9, no. 12 (1996): 3373-404. 
http://www.jstor.org/stable/26201460.

@author: Vivien Sainte Fare Garnot
"""

import pandas as pd
import numpy as np
import scipy as sp
import math
import copy


def ar1fit(x):
    xc=copy.copy(x)
    xc=xc-xc.mean()
    c0=c0hat(xc)
    c1=c1hat(xc)
    if c0>c1:
        g=gammat(xc)
        return g , sigmat(xc,g) , c0/(1-musquare(g,x.shape[0]))
    else:
        print('Non stationary value, setting gamma to 0.99')
        return 0.99 , sigmat(xc,0.99), c0/(1-musquare(g,x.shape[0]))

