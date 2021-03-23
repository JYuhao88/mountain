# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 08:53:43 2021

@author: jyh76
"""


import numpy as np

T = 10
tstep = 5e-3
tnum = int(1+T/tstep)
t = np.linspace(0, T, tnum)

delta = 5e-1
X = 2
xstep = 1e-2
xnum = int(1+2*X/xstep)
x = np.linspace(-X, X, xnum)

Y = X
ystep = xstep
ynum = xnum
y = x

xx, yy = np.meshgrid(x, y, indexing='ij')
d = 1/2
epsilon = 1e-8
solid = (np.abs(xx)==2) & (yy<=0)
solid = solid | ((np.abs(xx) > d) & (yy==0))
solid = solid | (yy==-2)

openbound_left = (xx==-X) & ((yy>0) & (yy != Y))
openbound_right = (xx==X) & (yy>0) & (yy != Y)
openbound_up = (yy==2)
openbound = openbound_left | openbound_right | openbound_up
oplr = openbound_left[0, :]
oplr_ = (yy>=0)
oplr_ = oplr_[0, :]

bound = solid | openbound
interior = ~ bound
outbound = (np.abs(xx)==X) | (np.abs(yy)==Y)

## initial
u = np.empty((tnum, xnum, ynum))
y0 = 0.5
r = np.sqrt(xx**2 + (yy-y0)**2)
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
    ux_diff = np.zeros((xnum, ynum))
    uy_diff = np.zeros((xnum, ynum))
    ux_diff[1:-1, :] = np.diff(u_last1, n=2, axis=0)
    uy_diff[:, 1:-1] = np.diff(u_last1, n=2, axis=1)
    u[i, interior] = (tstep**2)*(ux_diff[interior]/xstep**2 + uy_diff[interior]/ystep**2) \
                      + 2*u_last1[interior] - u_last2[interior]
    
    ## boundary
    u[i, solid] = 0
    
    ## x=-2 y>0
    tmp1 = (u[i, 1, oplr] - u_last1[1, oplr] + u_last1[0, oplr])/(xstep*tstep)
    tmp2 = (2*u_last1[0, oplr] - u_last2[0, oplr])/(tstep**2)
    tmp3 = np.diff(u[i, 0, oplr_], n=2)/(2*ystep**2)
    u[i, 0, oplr] = (tmp1+ tmp2 + tmp3)/(1/(tstep**2) + 1/(xstep*tstep))
    
    ## x=2 y>0
    tmp1 = (u[i, -2, oplr] + u_last1[-1, oplr] - u_last1[-2, oplr])/(tstep*xstep)
    tmp2 = (2*u_last1[-1, oplr]-u_last2[-1, oplr])/(tstep**2)
    tmp3 = np.diff(u[i, -1, oplr_], n=2)/(2*ystep**2)
    u[i, -1, oplr] = (tmp1+ tmp2 + tmp3)/(1/(tstep**2) + 1/(xstep*tstep))
    
    ## y=2
    tmp1 = (u[i, 1:-1, -2] + u_last1[1:-1, -1] - u_last1[1:-1, -2])/(tstep*ystep)
    tmp2 = (2*u_last1[1:-1, -1] - u_last2[1:-1, -1])/(tstep**2)
    tmp3 = np.diff(u_last1[:, -1], n=2)/(xstep**2)
    u[i, 1:-1, -1] = (tmp1+tmp2+tmp3)/(1/(tstep**2)+1/(ystep*tstep))

    # ## x = -2   y = 2
    # tmp1 = (u[i, 1, -1] - u_last1[1, -1] + u_last1[0, -1])/xstep
    # tmp2 = (u[i, 0, -2] + u_last1[0, -1] - u_last1[0, -2])/ystep
    # tmp3 = -3/2*(u[i, 0, -1] - 2*u_last1[0, -1] + u_last2[0, -1])/tstep
    # u[i, 0, -1] = (tmp1 + tmp2 + tmp3)/(1/xstep + 1/ystep)

    # ## x = 2   y = 2
    # tmp1 = (u[i, -2, -1] - u_last1[-2, -1] + u_last1[-1, -1])/xstep
    # tmp2 = (u[i, -1, -2] + u_last1[-1, -1] - u_last1[-1, -2])/ystep
    # tmp3 = -3/2*(u[i, -1, -1] - 2*u_last1[-1, -1] + u_last2[-1, -1])/tstep
    # u[i, -1, -1] = (tmp1 + tmp2 + tmp3)/(1/xstep + 1/ystep)
    
    u[i, 0, -1] = (u[i, 1, -1] + u[i, 0, -2])/2
    u[i, -1, -1] = (u[i, -1, -2] + u[i, -2, -1])/2
## plot

# import matplotlib.pyplot as plt
# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# ax.set_title('wave')
# for i in range(1000):
#     obsX = x
#     obsY = u[i,:]
#     ax.plot(obsX,obsY,c='b')
#     ax.view_init(azim=0, elev=90)
#     plt.pause(0.01)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation   
fig, ax = plt.subplots()
# u_tmp =u[0, :, 0:ynum//2]

img = plt.imshow(u[-1])

# nums = int(1/tstep)   

# def init():
#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)
#     return img

# def update(step):
#     z = u[step]
#     img.set_data(z) 
#     return img

# ani = FuncAnimation(fig, update, frames=range(nums), init_func=init,interval=5e3//nums)
plt.show()






