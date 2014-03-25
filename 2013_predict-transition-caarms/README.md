Project
=======

Prediction of psychotic transition from CAARMS items.
27 patient: 16 non-convertors, 11 convertors

Collaboration with SHU St-Anne, J Bourgin, MO Krebs

Abstract
=======

METHODS:
A supervised machine learning linear model based on the Elastic net logistic regression was used to distinguish between UHR-T and UHR-NT subjects. The Elastic net combines ridge (like SVM) and Lasso shrinkages. Lasso shrinkage promotes the selection of few relevant CAARMS items. The power of the classifier to generalize to an independent data set (predicting the clinical outcome given the all CAARMS items of unseen subjects) was assessed with a 10-fold stratified cross-validation. The significance of both prediction rates and selection of CAARMS items was assessed with random permutation.


Scripts
=======

mylib.py: I/O etc.
01_mulm.py : univariate analysis + features correlation
02_predict_svms.py : multivariate prediction
03_predict_logistic-lasso_and_assess_weights_variability.py: simi

Methods
=======

- Univariate filtering (P<0.05) + SVM => FAILED
- Sparse SVM => OK similar to logistic but has square loss, hard to refit on reduced dataset (without l1)
- Elasticnet Logistic regression (from ParsimonY) **CHOOSEN**

Tested datsets:
- CAARMS + PAS + Canabis => FAILED
- CAARMS => **CHOOSEN**


Results
=======

Univariate analysis (uncorrected t-test)
----------------------------------------

#        var     tstat      pval  df     mu0       mu1       sd0       sd1
#0    PAS2gr -2.865843  0.008314  25 -0.3750  0.636364  0.771389  0.771389
#2      @1.1 -2.452206  0.021525  25  2.5000  3.272727  0.616575  0.616575
#24     @7.6 -2.252298  0.033332  25  1.0000  2.545455  1.671343  1.671343
#16     @5.4 -2.197316  0.037482  25  1.5625  3.090909  1.621141  1.621141
#1   CB_EXPO -2.009756  0.055366  25  0.1250  0.818182  0.574960  0.574960
#18     @6.3 -1.797536  0.084341  25  0.2500  0.909091  1.083307  1.083307
#12     @4.3  1.750658  0.092269  25  3.8750  2.818182  1.748671  1.748671
#25     @7.7 -1.538554  0.136475  25  1.5000  2.454545  1.233151  1.233151
#22     @7.4  1.499319  0.146314  25  2.1250  1.000000  1.595448  1.595448
#6      @2.2 -1.495250  0.147367  25  1.2500  2.000000  1.206045  1.206045
#9      @3.3 -1.406208  0.171971  25  0.3125  0.818182  1.113404  1.113404
#3      @1.2 -1.236266  0.227849  25  1.3750  2.272727  1.813631  1.813631
#14     @5.2  1.147417  0.262072  25  3.5625  2.909091  1.564059  1.564059
#5      @2.1 -1.110598  0.277309  25  2.7500  3.181818  0.833196  0.833196
#11     @4.2  1.094649  0.284105  25  4.5000  4.000000  1.206045  1.206045
#13     @5.1  0.948344  0.352032  25  3.8125  3.181818  1.585054  1.585054
#19     @6.4 -0.903280  0.374999  25  2.3125  2.818182  1.192262  1.192262
#4      @1.3 -0.879334  0.387594  25  1.5625  2.090909  1.311110  1.311110
#20     @7.2  0.814706  0.422934  25  4.0000  3.545455  1.437399  1.437399
#17     @6.1 -0.759806  0.454477  25  0.7500  1.181818  1.465865  1.465865
#15     @5.3 -0.693770  0.494223  25  1.9375  2.363636  1.431638  1.431638
#8      @3.2 -0.614887  0.544186  25  1.5000  1.818182  1.266217  1.266217
#23     @7.5 -0.234327  0.816639  25  3.9375  4.090909  1.443137  1.443137
#10     @4.1 -0.196977  0.845438  25  1.6250  1.727273  1.212879  1.212879
#7      @3.1 -0.143957  0.886689  25  3.1250  3.181818  1.028519  1.028519
#26     @7.8 -0.079911  0.936945  25  2.7500  2.818182  2.036851  2.036851
#21     @7.3 -0.031192  0.975364  25  2.4375  2.454545  1.499311  1.499311

Multivariate prediction
-----------------------




OLDIES
======

Parmis les 27 11 ont fait la transition et 16 ne l'on pas faite
- Sensibilité (Taux de detection de les transitions)
72.72 % soit 8 / 11 (p = 0.03)

- Spécificité (Taux de detection de ceux qui n'ont pas transité ou 1 - Faux positifs)
87.5 % soit 14 / 16 (p = 0.01)

Nous avons un taux de bonne classification moyen de 81.4 %

Voici les items de la CAARMS qui interviennnent:
     coef   var
-0.084403  @4.3 anhédoni (symptome négatif) (t=1.75, p=0.092, univariate t-test)
 0.138800  @5.4 comportement agréssif dangereux (t=-2.19, p=0.037)
-0.113874  @7.4 labilité de l'humeur (t=1.49, p=0.146)
 0.063421  @7.6 trouble obsessionel et compulsif (t=-2.25, p=0.033332)
 0.011630  @7.7 symptomes dissociatifs (t=-1.53, p=0.136)

Signe positif + on est X moins on transite

4.3 et 7.2 très corrélés

la normalisation de Pre-Morbid Adjustment scale (PAS2gr) et de l'exposition au canabis (CB_EXPO) dégrade considérablement les résultats.

