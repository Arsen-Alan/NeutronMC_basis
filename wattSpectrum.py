#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:52:15 2019

@author: AlanArsen
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Watt constants for thermal fission of U-233

a = 0.977
b = 2.546
c = 1


# Functions definition 

watt = lambda var,paramA,paramB,paramC: paramC * np.exp(-var/paramA) * np.sinh(np.sqrt(paramB*var))

x = lambda var: var

x2 = lambda var: var**2

funOp = lambda var,paramA,paramB,paramC,fun1,fun2: fun1(var,paramA,paramB,paramC) * fun2(var)

normF = lambda paramA,paramB,paramC,fun : paramC/(quad(fun, 0, np.inf, args=(paramA,paramB,paramC))[0])


# Main

c = normF(a,b,c,watt) #Normalization

maxE = opt.fminbound(lambda x: -watt(x,a,b,c), 0, 20) # Most probable energy

meanE = quad(funOp, 0, np.inf, args=(a,b,c,watt,x))[0] # Mean energy

varE = quad(funOp, 0, np.inf, args=(a,b,c,watt,x2))[0] - meanE**2 # Variance

stdDvE = np.sqrt(varE) # Standard deviation

print("\nThe normalization constant C is %.5f" % c)
print("The most probable energy is %.5f MeV" % maxE)
print("The mean energy is %.5f MeV" % meanE)
print("The variance is %.5f MeV^2" % varE)
print("The standard deviation of E is %.5f MeV" % stdDvE)
    


x = np.linspace(0, 10, 1e4)

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.plot(x, watt(x,a,b,c))

# Major ticks every 20, minor ticks every 5
major_ticksx = np.arange(0, 11, 1)
minor_ticksx = np.arange(0, 11, 0.2)
major_ticksy = np.arange(0, 0.35, 0.05)
minor_ticksy = np.arange(0, 0.35, 0.01)

ax.set_xticks(major_ticksx)
ax.set_xticks(minor_ticksx, minor=True)
ax.set_yticks(major_ticksy)
ax.set_yticks(minor_ticksy, minor=True)

# And a corresponding grid
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

ax.set_xlabel('E[MeV]')
ax.set_ylabel('Relative probability')

plt.show()
