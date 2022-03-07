# -*- coding: utf-8 -*-
"""
ODESolver_Bad1.0
Last Modified: Jan 7 2022
@author: shugh
"""

import matplotlib.pyplot as plt  
import math as m  
import numpy as np 
import timeit 

'''
The Euler's forward method or explicit Euler's method:
    y[i+1] = y[i] + h * f(y[i],t[i]),
    where f(x[i], t[i]) is the differential equation evaluated
at y[i] and t[i].  Also applies the a system of ODEs.

This code is a "bad example" using a vectorized Euler is not good and the 
graphics are terrible (bad labels, small fonts, lines are rubbish, ... ). 
This is deliberate!
'''

#%% 

"Simple Euler ODE Solver"
def EulerForward(f,y,t,h): # Vectorized forward Euler (so no need to loop) 
# asarray converts to np array - so you can pass lists or numpy arrays
    k1 = h*np.asarray(f(y,t))                     
    y=y+k1
    return y 

"OBEs - with simple CW (harmonic) excitation"
def derivs(y,t): # derivatives function 
    dy=np.zeros((len(y))) 
    #dy = [0] * len(y) # could also use lists here which can be faster if 
                       # using non-vectorized ODE "
    dy[0] = 0.
    dy[1] = Omega/2*(2.*y[2]-1.)
    dy[2] = -Omega*y[1]
    return dy

#%%    
"Paramaters for ODEs - simple CW Rabi problem of coherent RWA OBEs"
Omega=2*np.pi # inverse time units, so when t=1, expect one full flop as that
              # would have an area of 2 pi 
dt = 0.001
tmax =1.
# numpy arrays for time and y ODE set
tlist=np.arange(0.0, tmax, dt) # gauarantees the same step size
npts = len(tlist)
y=np.zeros((npts,3)) # or can introduce arrays to append to
yinit = np.array([0.0,0.0,0.0]) # initial conditions (TLS in ground state)
y1=yinit # just a temp array to pass into solver
y[0,:]= y1

"Call ODE Solver"
start = timeit.default_timer()  # start timer for solver
for i in range(1,npts):   # loop over time
    y1=EulerForward(derivs,y1,tlist[i-1],dt) 
    y[i,:]= y1

# or
#   y[i,:]=EulerForward(derivs,y[i-1,:],tlist[i-1],dt) 
 

stop = timeit.default_timer()
print ("Time for Euler ODE Solver", stop - start) 

"Exact Solution for excited state population"
yexact = [m.sin(Omega*tlist[i]/2)**2 for i in range(npts)]

#%%
"GRAPHICS - way too simple so you must substantially improve!!"

# plot analytic and numerically obtained population n_e(t)
plt.plot(tlist, yexact, 'b')
plt.plot(tlist, y[:,2], 'r')

# This produces a horrible plot!
plt.legend(["Exact solution", 
            "Forward Euler"], loc='best')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.tight_layout()
plt.show() 

# Uncomment the following to save fig as pdf, but this will never work 
# for your LaTeX write up until you improve the graph!
#plt.savefig('./testOBE1.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')

