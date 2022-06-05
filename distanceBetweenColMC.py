#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:52:15 2019

@author: AlanArsen
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy.interpolate import interp1d
import scipy.optimize as opt
import matplotlib.pyplot as plt
import random
import os

# Watt constants for thermal fission of U-233
#
a = 0.977
b = 2.546
c = 1


# Numerical density of the fuel
#
numDens = 19.1*6.022e23/233 #cm-3

# Functions definition 
#
watt  = lambda var,paramA,paramB,paramC: paramC * np.exp(-var/paramA) * np.sinh(np.sqrt(paramB*var))
x     = lambda var: var
x2    = lambda var: var**2
funOp = lambda var,paramA,paramB,paramC,fun1,fun2: fun1(var,paramA,paramB,paramC) * fun2(var)
normF = lambda paramA,paramB,paramC,fun: paramC/(quad(fun, 0, np.inf, args=(paramA,paramB,paramC))[0])

probF = lambda delta, sigma: erf(delta/(sigma * np.sqrt(2)))

def findXS(argE, argDataXS):
    
    i = 0
    
    while argE > argDataXS[i,0]:
        
        i += 1
        
    interp = interp1d(data[i-1:i+1,0], data[i-1:i+1,1])

    return interp(argE)
        
sortS = lambda argXS, numDensU: -(1/(argXS*numDensU)) * np.log(random.random())

def singleSim(argSimNum, argEnergyScalling, argProbScalling, argDataXS,numDensU):
    
    i = 0
    fcount = 0
    fSumE  = 0
    fSumE2 = 0
    fSumS  = 0
    fSumS2 = 0
    
    while i<int(argSimNum):
        
        fcount = fcount + 1
        E = argEnergyScalling * random.random()
        P = argProbScalling * random.random()
        
        if watt(E,a,b,c) >=  P:
        
            i = i + 1
            s = sortS(findXS(E*1e6, argDataXS)*1e-24,numDensU)
            
            fSumE  += E
            fSumE2 += E**2
            fSumS  += s
            fSumS2 += s**2
            
    return fSumE, fSumE2, fcount, fSumS, fSumS2

# Deterministic calculations
#
c = normF(a,b,c,watt) #Normalization
maxEdet   = opt.fminbound(lambda x: -watt(x,a,b,c), 0, 20) # Most probable energy
meanEdet  = quad(funOp, 0, np.inf, args=(a,b,c,watt,x))[0] # Mean energy
varEdet   = quad(funOp, 0, np.inf, args=(a,b,c,watt,x2))[0] - meanEdet**2 # Variance
stdDvEdet = np.sqrt(varEdet) # Standard deviation

# Main
#
print("\nRunning Monte Carlo simulation...")

path = "U-233TotalXS.txt"
path = os.path.join(os.getcwd(),path)

f = open(path,"r")
data = f.readlines()

for i in range(0, len(data)): #formating the data from text to two float items per line (E and xs)
    
    data[i] = data[i].split()
    for j in range(0,len(data[i])):
        data[i][j] = float(data[i][j])
        
data = np.array(data) # converting to numpy array for easier manipulation

# Main
#
simNum = 1e5
energyScalling = 20 # MeV
probScalling = watt(maxEdet,a,b,c)

sumE, sumE2, count, sumS, sumS2 = singleSim(simNum, energyScalling, probScalling, data, numDens)
    
meanE = sumE/simNum
stDvMeanE = np.sqrt((sumE2/simNum - meanE**2)/simNum)
deltaE = 2 * stDvMeanE
pE = probF(deltaE,stDvMeanE) # Confidence interval

meanS = sumS/simNum
stDvMeanS = np.sqrt((sumS2/simNum - meanS**2)/simNum)
deltaS = 2 * stDvMeanS
pS = probF(deltaS,stDvMeanS) # Confidence interval

eff = simNum/count


print("\n.....MONTE CARLO RESULTS.....")

print("\nMean E: %.3f MeV" % meanE, "+- %.3f" % deltaE)
print("Confidence interval: %.3f" %pE)
print("Variance of the mean value: %.5f MeV^2" % stDvMeanE**2)

print("\nMean S: %.3f cm" % meanS, "+- %.3f" % deltaS)
print("Confidence interval: %.3f" %pS)
print("Variance of the mean value: %.5f cm^2" % stDvMeanS**2)

print("Efficiency of the sampling method is: %.3f " %eff)

## XS plot
##
#x = np.linspace(0, 2e8, 1e6)

#fig = plt.figure()

#ax = fig.add_subplot(1, 1, 1)
#ax.loglog(data[:,0], data[:,1])

#ax.grid(b=True, which='both', color='0.65', linestyle='-')

#ax.set_xlabel('E [eV]')
#ax.set_ylabel('Total microscopic cross-section [b]')

#plt.show()
