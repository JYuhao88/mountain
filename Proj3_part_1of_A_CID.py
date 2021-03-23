# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 18:10:11 2021

@author: jyh76
"""

import numpy as np

T = 10
tstep = 1e-3
tnum = int(T/tstep)
t = np.linspace(0, T, tnum)

delta = 5e-1
X = 2
xstep = 1e-2
xnum = int(2*X/xstep)
x = np.linspace(-X, X, xnum)

## initial
c = 4
nu = c*tstep/xstep
u = np.empty((tnum, xnum))
delta_in = (abs(x)<=delta)
delta_out = ~delta_in

u[0, delta_in] = np.cos(np.pi*x[delta_in]/(2*delta))
u[0, delta_out] = 0

u[1, 0] = u[0, 0]
u[1, -1] = u[0, -1]
diff = np.diff(u[0, :], n=2, axis=0)
u[1,1:-1] = u[0,1:-1] + nu**2/2*(diff)

## 
for i in range(2, tnum):
    u_last1 = u[i-1, :]
    u_last2 = u[i-2, :]
    diff = np.diff(u[i-1, :], n=2, axis=0)
    u[i, 1:-1] = ((tstep**2)*(c**2)*(diff)/(xstep**2)) \
        + 2*u_last1[1:-1] - u_last2[1:-1]
    
    ## boundary
    ## a
    # u[i, 0] = (c*u[i, 1]/xstep + u_last1[0]/tstep)/(c/xstep + 1/tstep)
    # u[i, -1] =  (c*u[i, -2]/xstep + u_last1[-1]/tstep)/(c/xstep + 1/tstep)
    
    ## b
    u[i, 0] = 0
    u[i, -1] = 0

## plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation   

fig, ax = plt.subplots() 
line, = plt.plot([], [], '.-',color='green')
nums = int(1/tstep)   

def init():
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    return line

def update(step):
    y =  u[step, :] 
    line.set_data(x, y) 
    return line

ani = FuncAnimation(fig, update, frames=nums, \
                    init_func=init,interval=5e3//nums)
plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    