# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 08:53:43 2021

@author: jyh76
"""


import numpy as np

T = 10
tstep = 1e-2
tnum = int(T/tstep)
t = np.linspace(0, T, tnum)

delta = 5e-1
X = 2
xstep = 1e-2
xnum = int(2*X/xstep)
x = np.linspace(-X, X, xnum)

Y = X
ystep = xstep
ynum = xnum
y = x

xx, yy = np.meshgrid(x, y, indexing='xy')
epsilon = 1e-8
solid = np.abs(xx)==2 