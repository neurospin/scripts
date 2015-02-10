# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 09:40:13 2015

@author: cp243490
"""

import numpy as np

import matplotlib.pyplot as plt


# plot accuracy for each ROI
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

N_rois = 13  # number of ROIs
accuracy = [0.8636363636, 0.8484848485, 0.8333333333, 0.8181818182,
            0.8181818182, 0.803030303, 0.7575757576, 0.7424242424,
            0.6515151515, 0.6363636364, 0.6212121212, 0.803030303,
            0.8181818182]
accuracy_std = [0.0617454452, 0.1131923142, 0.0883883476, 0.0903696114,
                 0.0880340843, 0.0679562768, 0.0748065556, 0.0601502748,
                 0.1020603068, 0.0765191211, 0.0821478539, 0.0614636297,
                 0.0674948558]
specificity = [0.935483871, 0.8064516129, 0.8387096774, 0.8709677419,
               0.7741935484, 0.8064516129, 0.7741935484, 0.7419354839,
               0.6129032258, 0.7419354839, 0.7096774194, 0.8064516129,
               0.8387096774]
sensitivity = [0.8, 0.8857142857, 0.8285714286, 0.7714285714, 0.8571428571,
               0.8, 0.7428571429, 0.7428571429, 0.6857142857, 0.5428571429,
               0.5428571429, 0.8, 0.8]

ind = np.arange(N_rois)                # the x locations for the groups
width = 0.35

rects = ax2.bar(ind, accuracy, width * 2,
                color='orange',
                yerr=accuracy_std,
                error_kw=dict(elinewidth=2, ecolor='black'))

rects1 = ax1.bar(ind, specificity, width, color='blue')
rects2 = ax1.bar(ind + width, sensitivity, width, color='green')

ax1.set_xlim(-width, len(ind) + width)
ax1.set_ylim(0, 1)
ax1.set_ylabel('Probability')
ax1.set_title('Specificity and sensitivity by region of interest')
ax2.set_xlim(-width, len(ind) + width)
ax2.set_ylim(0, 1)
ax2.set_ylabel('Probability')
ax2.set_title('Accuracy by region of interest')
xTickMarks = ["Hippocampe", "Cyngulate Gyrus, Ant", "Frontal Pole",
              "Middle Frontal Gyrus", "Superior Frontal Gyrus",
              "Frontal Orbital Cortex", "Insula", "Putamen", "Caudate",
              "Frontal Medial Cortex", "Amygdala", "maskdep", "brain"]
ax1.set_xticks(ind + width)
ax2.set_xticks(ind+width)
xtickNames1 = ax1.set_xticklabels(xTickMarks)
xtickNames2 = ax2.set_xticklabels(xTickMarks)
plt.setp(xtickNames1, rotation=45, fontsize=10)
plt.setp(xtickNames2, rotation=45, fontsize=10)

ax1.legend((rects1[0], rects2[0]), ('Specificity', 'Sensitivity'))
ax2.legend((rects[0],), ('Accuracy',))

plt.show()