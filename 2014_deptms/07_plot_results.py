# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 09:40:13 2015

@author: cp243490
"""

import numpy as np

import matplotlib.pyplot as plt
import os
import pandas as pd

BASE_PATH = "/neurospin/brainomics/2014_deptms"

MOD = "MRI"

ENETTV_PATH = os.path.join(BASE_PATH, "results_enettv")
SVM_PATH = os.path.join(BASE_PATH, "results_svm")
score_enettv_file = os.path.join(ENETTV_PATH,
                          "Summary_results_enettv_dCV_corrected_pvalues.csv")

scores_enettv = pd.read_csv(score_enettv_file)
score_svm_file = os.path.join(SVM_PATH, 'svm_scores.csv')
scores_svm = pd.read_csv(score_svm_file)

# PLOT: scores obtained with 10*10 double cross validation +
# enettv logistic regression VS scores obtained with svm algo
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)

N_rois = 13  # number of ROIs

enettv_accuracy = scores_enettv.accuracy.values.tolist()
enettv_accuracy_std = scores_enettv.accuracy_std.values.tolist()
enettv_specificity = scores_enettv.specificity.values.tolist()
enettv_sensitivity = scores_enettv.sensitivity.values.tolist()

svm_accuracy = scores_svm.accuracy.values.tolist()
svm_specificity = scores_svm.specificity.values.tolist()
svm_sensitivity = scores_svm.sensitivity.values.tolist()

ind = np.arange(N_rois)                # the x locations for the groups
width = 0.35

rects_enettv = ax2.bar(ind, enettv_accuracy, width * 2,
                color='orange',
                yerr=enettv_accuracy_std,
                error_kw=dict(elinewidth=2, ecolor='black'))
rects_svm = ax4.bar(ind, svm_accuracy, width * 2,
                color='orange',
                error_kw=dict(elinewidth=2, ecolor='black'))

rects1 = ax1.bar(ind, enettv_specificity, width, color='blue')
rects2 = ax1.bar(ind + width, enettv_sensitivity, width, color='green')
rects3 = ax3.bar(ind, svm_specificity, width, color='blue')
rects4 = ax3.bar(ind + width, svm_sensitivity, width, color='green')

ax1.set_xlim(-width, len(ind) + width)
ax1.set_ylim(0, 1)
ax1.set_ylabel('Probability')
ax1.set_title('TV-ElasticNet Specificity and sensitivity by region of interest')
ax2.set_xlim(-width, len(ind) + width)
ax2.set_ylim(0, 1)
ax2.set_ylabel('Probability')
ax2.set_title('TV-ElasticNet Accuracy by region of interest')

ax3.set_xlim(-width, len(ind) + width)
ax3.set_ylim(0, 1)
ax3.set_ylabel('Probability')
ax3.set_title('SVM Specificity and sensitivity by region of interest')
ax4.set_xlim(-width, len(ind) + width)
ax4.set_ylim(0, 1)
ax4.set_ylabel('Probability')
ax4.set_title('SVM Accuracy by region of interest')

xTickMarks_enettv = scores_enettv.ROI.values.tolist()
xTickMarks_svm = scores_svm.ROI.values.tolist()

ax1.set_xticks(ind + width)
ax2.set_xticks(ind + width)
ax3.set_xticks(ind + width)
ax4.set_xticks(ind + width)

xtickNames1 = ax1.set_xticklabels(xTickMarks_enettv)
xtickNames2 = ax2.set_xticklabels(xTickMarks_enettv)
xtickNames3 = ax3.set_xticklabels(xTickMarks_svm)
xtickNames4 = ax4.set_xticklabels(xTickMarks_svm)

plt.setp(xtickNames1, rotation=45, fontsize=10)
plt.setp(xtickNames2, rotation=45, fontsize=10)
plt.setp(xtickNames3, rotation=45, fontsize=10)
plt.setp(xtickNames4, rotation=45, fontsize=10)

ax1.legend((rects1[0], rects2[0]), ('Specificity', 'Sensitivity'))
ax2.legend((rects_enettv[0],), ('Accuracy',))
ax3.legend((rects3[0], rects4[0]), ('Specificity', 'Sensitivity'))
ax4.legend((rects_svm[0],), ('Accuracy',))

fig1.savefig(os.path.join(ENETTV_PATH, "recall_scores_diagram"))
fig2.savefig(os.path.join(ENETTV_PATH, "accuracy_scores_diagram"))
fig3.savefig(os.path.join(SVM_PATH, "recall_scores_diagram"))
fig4.savefig(os.path.join(SVM_PATH, "accuracy_scores_diagram"))

plt.show()