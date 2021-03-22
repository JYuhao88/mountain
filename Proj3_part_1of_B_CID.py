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
d = 1/2
epsilon = 1e-8
solid = (np.abs(xx)==2) & (yy<=0)
solid = solid | ((np.abs(xx) > d) & (yy==0))
solid = solid | (y==-2)

openbound = (np.abs(xx)==2) & (yy>0)
openbound = openbound | (yy==2)
bound = solid | openbound
interior = ~ bound
outbound = (np.abs(xx)==X) | (np.abs(yy)==Y)

## initial
u = np.empty((tnum, xnum, ynum))
y0 = 0.5
r = np.sqrt(xx**2 + (y-y0)**2)
delta_in = r<=delta
delta_out = ~ delta_in 
u[0, delta_in] = np.cos(np.pi*r[delta_in]/(2*delta))
u[0, delta_out] = 0

u[1, bound] = u[0, bound]
u0x_diff = np.empty((xnum, ynum))
u0y_diff = np.empty((xnum, ynum))
u0x_diff[1:-1, :] = np.diff(u[0], n=2, axis=0)
u0y_diff[:, 1:-1] = np.diff(u[0], n=2, axis=1)

u[1, interior] = u[0, interior] + tstep**2/2*(u0x_diff[interior]/(xstep**2) + u0y_diff[interior]/(ystep**2))

for i in range(2, tnum):
    u_last1 = u[i-1]
    u_last2 = u[i-2]
    ux_diff = np.empty((xnum, ynum))
    uy_diff = np.empty((xnum, ynum))
    ux_diff[1:-1, :] = np.diff(u_last1, n=2, axis=0)
    uy_diff[:, 1:-1] = np.diff(u_last1, n=2, axis=1)
    u[i, interior] = ((tstep**2)*(ux_diff[interior]/xstep**2 + uy_diff[interior]/ystep**2) \
                      + 2*u_last1[interior] - u_last2[interior]



















