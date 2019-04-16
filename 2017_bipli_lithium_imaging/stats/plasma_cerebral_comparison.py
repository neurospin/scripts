# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:50:43 2019

@author: JS247994
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import Model
from scipy import stats
from matplotlib.pyplot import figure

plasmatic_Li=np.array([0.77, 0.95, 0.77, 1.08, 0.61, 0.73, 0.8, 0.69, 0.9, 0.52, 0.85, 0.61, 0.87, 0.71, 0.46, 0.7, 1.09, 0.67, 0.92, 0.8, 0.93 ])
mean_brain_Li=np.array([0.319729721, 0.300374628, 0.275451928, 0.359628966, 0.223182271, 0.27369042, 0.233443453, 0.324570566, 0.287107272, 0.158162935, 0.301351724, 0.194404482, 0.378551915, 0.201839826, 0.152508693, 0.180311263, 0.337347153, 0.192206623, 0.324701602, 0.241411618, 0.31358783])
mean_brain_Li=np.array([0.41,0.36,0.34,0.41,0.28,0.33,0.30,0.38,0.35,0.24,0.36,0.26,0.44,0.26,0.24,0.25,0.39,0.26,0.38,0.32,0.37])
max_brain_Li=np.array([0.916769719, 0.739344027, 0.706912103, 0.966450641, 0.640178574, 0.664865945, 0.632191736, 0.800659063, 0.845194116, 0.46973291, 0.628303665, 0.593717277, 0.815920188, 0.539296917, 0.513865226, 0.423221457, 0.672369047, 0.462600865, 0.645531385, 0.61183153, 0.676899518])
max_brain_Li=np.array([1.18,1.1,1.18,1.4,0.92,0.96,0.89,1.22,0.90,0.78,1.0,0.83,1.08,0.82,0.69,0.70,1.01,0.80,0.99,1.00,0.99])
#max_brain_Li=mean_brain_Li
#max_brain_Li=np.array([1.178288978, 1.105146417, 1.178931973, 0.966450641, 0.925926799, 0.962055589, 0.889352121, 1.222705362, 0.896689229, 0.772725067, 1.001959621, 0.839322761, 1.08463172, 0.822356757, 0.690282021, 0.697704063, 1.010515726, 0.797145596, 0.991680935, 1.004056329, 0.992975985])



font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

fig=figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(1,1,1)

plt.scatter(plasmatic_Li, mean_brain_Li,facecolors='none', edgecolors='r', label='data')

slope, intercept, r_value, p_value, std_err = stats.linregress(plasmatic_Li,mean_brain_Li)
xval=np.arange(0,1.2,0.1)
#plt.plot(plasmatic_Li,slope*plasmatic_Li+intercept, 'r--')

degrees = [1]       # list of degrees of x to use
matrix = np.stack([plasmatic_Li**d for d in degrees], axis=-1)   # stack them like columns
coeff = np.linalg.lstsq(matrix, mean_brain_Li)[0]  

explained_var=np.sum(np.square(plasmatic_Li*coeff-mean_brain_Li))
total_var=np.sum(np.square(max_brain_Li-np.mean(mean_brain_Li)))
Rsquared=1-(explained_var/total_var)

ax1.plot(xval,slope*xval+intercept, 'r:')
plt.xlabel('Plasmatic [Li] (in mmol/L)',fontdict=font)
plt.ylabel('Average cerebral [Li] (in mmol/L)',fontdict=font)
percent=slope*100
percentstring="%.0f" % percent
plottextslope=r''+str(percentstring)+'% of concentration'
r_value_string="%.2f" % r_value
plottestr=r'R'+chr(0x00B2)+' = '+r_value_string
#ax1.text(0.25,0.36,plottextslope)
ax1.set_xlim([0,1.15])
ax1.set_ylim([0,0.5])
#plt.gca().set_aspect('equal', adjustable='box')
#ax1.text(0.25,0.32,plottestr)
ax1.grid(True)
plt.show()

fig=figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
ax2 = fig.add_subplot(1,1,1)
plt.scatter(plasmatic_Li,max_brain_Li, facecolors='none', edgecolors='b', label='data')
#slope, intercept, r_value, p_value, std_err = stats.linregress(plasmatic_Li,max_brain_Li)
degrees = [1]       # list of degrees of x to use
matrix = np.stack([plasmatic_Li**d for d in degrees], axis=-1)   # stack them like columns
coeff = np.linalg.lstsq(matrix, max_brain_Li)[0]  
ax2.plot(xval,coeff*xval, 'b:')
#plt.plot(xval,slope*xval+intercept, 'b:')

explained_var=np.sum(np.square(plasmatic_Li*coeff-max_brain_Li))
total_var=np.sum(np.square(max_brain_Li-np.mean(max_brain_Li)))
Rsquared=1-(explained_var/total_var)

plt.xlabel('Plasmatic [Li] (in mmol/L)',fontdict=font)
plt.ylabel('Max cerebral [Li] (in mmol/L)',fontdict=font)
percent=coeff*100
percentstring="%.0f" % percent
plottextslope=r''+str(percentstring)+'% of concentration'
#ax2.text(0.25,0.92,plottextslope)
r_value_string="%.2f" % Rsquared
plottestr=r'R'+chr(0x00B2)+' = '+r_value_string
#ax2.text(0.25,0.82,plottestr)
ax2.grid(True)
ax2.set_xlim([0,1.15])
ax2.set_ylim([0,1.5])
#plt.gca().set_aspect('equal', adjustable='box')
plt.show()


degrees = [1]       # list of degrees of x to use
matrix = np.stack([plasmatic_Li**d for d in degrees], axis=-1)   # stack them like columns
coeff = np.linalg.lstsq(matrix, max_brain_Li)[0]  
explained_var=np.sum(np.square(plasmatic_Li*coeff-max_brain_Li))
total_var=np.sum(np.square(max_brain_Li-np.mean(max_brain_Li)))
Rsquared=np.sqrt(1-(explained_var/total_var))
#plt.plot(xval,coeff*xval)
#from matplotlib.pyplot import figure

#fig=figure(num=None, figsize=(20, 6), dpi=80, facecolor='w', edgecolor='k')
#ax1 = fig.add_subplot(1,1,1)

#plt.gca().set_aspect('equal', adjustable='box')