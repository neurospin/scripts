# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ce script temporaire est sauvegardé ici :
/home/ad247405/.spyder2/.temp.py
"""

#TOWARD_ON SIMILARITY COEFFICIENTS: TV AND SVM
coef=np.load(os.path.join(BASE_PATH,'toward_on','Logistic_L1_L2_TV_with_HC','betas_subj.npy'))

coef=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','betas_subj.npy'))

#ON/OFF SIMILARITY COEFFICIENTS: TV AND SVM
coef=np.load(os.path.join(BASE_PATH,'results','Logistic_L1_L2_TV_withHC','betas_subj.npy'))

coef=np.load(os.path.join(BASE_PATH,'results','svm_with_HC','betas_subj.npy'))



from brainomics import array_utils 
from statsmodels.stats.inter_rater import fleiss_kappa

#threshold betas to compute fleiss_kappa and DICE
coef_t = np.vstack([array_utils.arr_threshold_from_norm2_ratio(coef[i, :], .99)[0] for i in xrange(coef.shape[0])])
# Compute fleiss kappa statistics


coef_signed = np.sign(coef_t)
table = np.zeros((coef_signed.shape[1], 3))
table[:, 0] = np.sum(coef_signed == 0, 0)
table[:, 1] = np.sum(coef_signed == 1, 0)
table[:, 2] = np.sum(coef_signed == -1, 0)
fleiss_kappa_stat = fleiss_kappa(table)
print fleiss_kappa_stat


# Paire-wise Dice coeficient

coef_t = np.vstack([array_utils.arr_threshold_from_norm2_ratio(coef[i, :], .99)[0] for i in xrange(coef.shape[0])])
coef_signed = np.sign(coef_t)
coef_n0 = coef_t!= 0
ij = [[i, j] for i in xrange(23) for j in xrange(i+1, 23)]
#print [[idx[0], idx[1]] for idx in ij]
dice_bar = np.mean([float(np.sum(coef_signed[idx[0], :] == coef_signed[idx[1], :])) /\
(np.sum(coef_n0[idx[0], :]) + np.sum(coef_n0[idx[1], :]))for idx in ij])

print dice_bar