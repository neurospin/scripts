# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:43:41 2019

@author: JS247994
"""

import numpy as np
from scipy import stats

Totalblood=np.array([0.73,0.37,0.75,0.72,0.42,0.51,0.34,0.42,0.8,0.86,0.48,0.47,0.71])
Plasmablood=np.array([0.9,0.52,0.87,0.85,0.61,0.71,0.46,0.67,1.09,0.92,0.8,0.7,0.93])

slope, intercept, r_value, p_value, std_err = stats.linregress(Totalblood,Plasmablood)

Totalblood_group1=np.array([0.58,0.92,0.32,0.41,0.78,0.54,0.33,0.49,0.58,0.61])

Plasmablood_group1=slope*Totalblood_group1+intercept