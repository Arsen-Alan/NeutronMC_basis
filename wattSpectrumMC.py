#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:52:15 2019

@author: AlanArsen
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import erf
import scipy.optimize as opt
import random

# Watt constants for thermal fission of U-233
#
a = 0.977
b = 2.546
c = 1


# Functions definition 
#
watt = lambda var,paramA,paramB,paramC: paramC * np.exp(-var/paramA) * np.sinh(np.sqrt(paramB*var))
x = lambda var: var
x2 = lambda var: var**2
funOp = lambda var,paramA,paramB,paramC,fun1,fun2: fun1(var,paramA,paramB,paramC) * fun2(var)
normF = lambda paramA,paramB,paramC,fun: paramC/(quad(fun, 0, np.inf, args=(paramA,paramB,paramC))[0])

probF = lambda delta, sigma: erf(delta/(sigma * np.sqrt(2)))

def singleSim(argSimNum, argEnergyScalling, argProbScalling):
    
    fcount = 0
    i = 0
    fSumE = 0
    fSumE2 = 0
    
    while i<int(argSimNum):
    
        fcount = fcount + 1
    
        E = argEnergyScalling * random.random()
        P = argProbScalling * random.random()
    
        #ax.scatter(E, P, marker='o',s=5) #For this to work Spectrum plot must be before Main
    
        if watt(E,a,b,c) >=  P:
        
            i = i + 1
        
            fSumE = fSumE + E
            fSumE2 = fSumE2 + E**2
    
    return fSumE, fSumE2, fcount

# Deterministic calculations
#
c = normF(a,b,c,watt) #Normalization
maxEdet = opt.fminbound(lambda x: -watt(x,a,b,c), 0, 20) # Most probable energy
meanEdet = quad(funOp, 0, np.inf, args=(a,b,c,watt,x))[0] # Mean energy
varEdet = quad(funOp, 0, np.inf, args=(a,b,c,watt,x2))[0] - meanEdet**2 # Variance
stdDvEdet = np.sqrt(varEdet) # Standard deviation

# Main
#
print("\nRunning Monte Carlos simulation...")

simNum = 1e5
energyScalling = 30 # MeV
probScalling = watt(maxEdet,a,b,c)

runNum = 1e3
acceptedN = 0

for i in range(int(runNum)):
    
    print(i+1)
    
    sumE, sumE2, count = singleSim(simNum, energyScalling, probScalling)
    
    meanE = sumE/simNum
    stDvMeanE = np.sqrt((sumE2/simNum - meanE**2)/simNum)
    delta = 2 * stDvMeanE
    
    if meanE+delta > meanEdet and meanE-delta < meanEdet:
        acceptedN = acceptedN + 1

varE = sumE2/simNum - meanE**2
stDvE = np.sqrt(varE)
varMeanE = varE/simNum

p = probF(delta,stDvMeanE) # Confidence interval
eff = simNum/count
# eff2 = 1/(probScalling*energyScalling)
accepPorc = acceptedN/runNum

print("\n.....MONTE CARLO RESULTS.....")

print("\nResult: %.3f MeV" % meanE, "+- %.3f" % delta)
print("Confidence interval: %.3f" %p)
print("Variance of the mean value: %.5f MeV^2" % varMeanE)
print("Efficiency of the sampling method is: %.3f " %eff)
print("Real expectation value inside the confidence interval %.3f" %accepPorc)

print("\n\n.....DETERMINISTIC RESULTS.....")

print("\nAverage energy: %.3f MeV" % meanEdet)
print("Variance: %.3f MeV^2" % varEdet)
print("Standard deviation: %.3f MeV" % stDvE)

    
#plt.show()

## Spectrum plot
##
#x = np.linspace(0, 10, 1e4)
#
#fig = plt.figure()
#
#ax = fig.add_subplot(1, 1, 1)
#ax.plot(x, watt(x,a,b,c))
#
## Major and minor ticks
#major_ticksx = np.arange(0, 11, 1)
#minor_ticksx = np.arange(0, 11, 0.2)
#major_ticksy = np.arange(0, 0.35, 0.05)
#minor_ticksy = np.arange(0, 0.35, 0.01)
#
#ax.set_xticks(major_ticksx)
#ax.set_xticks(minor_ticksx, minor=True)
#ax.set_yticks(major_ticksy)
#ax.set_yticks(minor_ticksy, minor=True)
#
## And a corresponding grid
#ax.grid(which='minor', alpha=0.2)
#ax.grid(which='major', alpha=0.5)
#
#ax.set_xlabel('E[MeV]')
#ax.set_ylabel('Relative probability')
#
##plt.show()