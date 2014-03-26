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

elastic net logistic regression
Lasso + ridge
Stratified CV
Permutation

Prediction accuracy scores
Coefficients
The the Lasso shrinkage comprises in the elastic net logistic regression promotes the selection of few items among the 25 possible items.

Results
=======
The elastic net logistic regression achieved highly significant prediction of the conversion given the 25 CAARMS items of independent set of subjects. Using the ten fold stratified cross-validation the classification achieved 81.8% (P = 0.035) of sensitivity (correct prediction of UHR-T subjects) and 93.7% (P = 0.0016) of specificity (correct prediction of UHR-NT subjects). All other binary classification scores were also highly significant: positive predictive value (90%, P = 0.0002), negative predictive value (88%, P = 0.0062), area under curve (0.83, P = 0.0086) and F1 scores on both UHR-T (0.857, P = 0.0004) and UHR-NT (0.90, P = 0.0002) subjects.


The whole 10 fold cross-validation round led to the selection of only 6 different CAARMS items, and noticeably four of them () where systematically selected across the ten folds. This suggest a stable and reproducible predictive patterns that replicate over the cross-validation re-sampling. 

and 

Four CAARMS items were always selected : 

Multivariate prediction
-----------------------

Full data
~~~~~~~~~~

Predictions:
score	    val	            p-val
prec_0	    0.8823529412	0.0062
prec_1	    0.9	            0.0002
prec_mean	0.8911764706	0.0003
recall_0	0.9375        	0.0016
recall_1	0.8181818182	0.0359
recall_mean	0.8778409091	0.0004
f_0	        0.9090909091	0.0002
f_1	        0.8571428571	0.0004
f_mean	    0.8831168831	0.0004
support_0	16	            1
suppor_1	11	            1
auc	        0.8352272727	0.0086

Coefficients:
count	count_pval	mean	        mean_pval	var	    z	            z_pval
10	    1	        -0.0330010508	0.9633	    inter	-0.1661938279	0.9416
4	    0.2665	    0.0060310892	0.3953	    @1.2	0.6090562234	0.2548
10	    0.0381	    -0.1879307305	0.0267	    @4.3	-3.8547418439	0.0062
10	    0.0707	    0.2861784479	0.0123	    @5.4	3.6504777519	0.0169
10	    0.1018	    -0.2486472376	0.0274	    @7.4	-5.2497576126	0.0103
10	    0.0715	    0.1522711267	0.0664	    @7.6	1.8131611685	0.0836
7	    0.158	    0.0523902029	0.1765	    @7.7	1.0775088485	0.1284



Reduced data
~~~~~~~~~~~~

Predictions:
prec_0	    0.8823529412	0.001
prec_1	    0.9	            0
prec_mean	0.8911764706	0
recall_0	0.9375	        0.0001
recall_1	0.8181818182	0.0159
recall_mean	0.8778409091	0.0001
f_0	        0.9090909091	0
f_1	        0.8571428571	0.0001
f_mean	    0.8831168831	0.0001
support_0	16	            1
suppor_1	11	            1
auc	        0.9375	        0

Coefficients:
count	count_pval	mean	mean_pval	var	z	z_pval
10	1	2.1441287709	0.1108	inter	5.1584028117	0.0464
10	1	-1.2047445808	0.007	@4.3	-15.0787477839	0.0001
10	1	1.4774808887	0.0019	@5.4	9.7615092629	0.0025
10	1	-1.7931352757	0.0006	@7.4	-13.1761157949	0.0004
10	1	1.0345555593	0.0111	@7.6	5.236217604	0.0377


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

