# -*- coding: utf-8 -*-
"""
Created on Thu May 26 10:33:04 2016

@author: ad247405
"""

import matplotlib.pyplot as plt
import scipy
import scipy.interpolate
from scipy.interpolate import Rbf

a = np.load('/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/time/tv_1e-6_1e-1/nipals_ite_for_comp:2.npz')

eps_0 = a['func'][0] - a['func'][0][-1]
eps_1 = a['func'][1] - a['func'][1][-1]
eps_2 = a['func'][2] - a['func'][2][-1]

time_0 = a['time'][0]
time_1 = a['time'][1]
time_2 = a['time'][2]

plt.plot(np.cumsum(time_0),np.abs(eps_0))
plt.xlabel('Time')
plt.ylabel('f - f*')
plt.title('component 0')
plt.yscale('log')

plt.plot(np.cumsum(time_1),np.abs(eps_1))
plt.xlabel('Time')
plt.ylabel('f - f*')
plt.title('component 1')
plt.yscale('log')

plt.plot(np.cumsum(time_2),np.abs(eps_2))
plt.xlabel('Time')
plt.ylabel('f - f*')
plt.title('component 2')
plt.yscale('log')


#Interpolate to find time necesarry to achieve a given precision
#########################################################################################

ius_0 = Rbf(np.abs(eps_0),np.cumsum(time_0),smooth=0.1)
x= np.linspace(0,np.abs(eps_0).max())
y = ius_0(x)
plt.plot(np.abs(eps_0),np.cumsum(time_0),'o',x,y)


ius_1 = Rbf(np.abs(eps_1),np.cumsum(time_1),smooth=0.1)
x= np.linspace(0,np.abs(eps_1).max())
y = ius_1(x)
plt.plot(np.abs(eps_1),np.cumsum(time_1),'o',x,y)


ius_2 = Rbf(np.abs(eps_2),np.cumsum(time_2),smooth=0.8)
x= np.linspace(0,np.abs(eps_2).max())
y = ius_2(x)
plt.plot(np.abs(eps_2),np.cumsum(time_2),'o',x,y)


print "Total time necessary in order to achieve a precision of 1e-3;", ius_0(1e-03) + ius_1(1e-03) + ius_2(1e-03) 
print "Total time necessary in order to achieve a precision of 1e-4;", ius_0(1e-04) + ius_1(1e-04) + ius_2(1e-04) 
print "Total time necessary in order to achieve a precision of 1e-5;", ius_0(1e-05) + ius_1(1e-05) + ius_2(1e-05) 
print "Total time necessary in order to achieve a precision of 1e-6;", ius_0(1e-06) + ius_1(1e-06) + ius_2(1e-06) 


#########################################################################################
