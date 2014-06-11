12_cv_multivariate_BMI_image.py: 10-fold cross-validation "by hand" and computation of R2 value for various sets (alpha, l1_ratio), where alpha and l1_ratio are defined by two lists respectively, using ElasticNet or Parsimony

12-1_cv_multivariate_BMI_image_fast.py: 10-fold cross-validation "by hand" and computation of R2 value for various sets (alpha, l1_ratio), where alpha is fixed and l1_ratio is defined in a list, using ElasticNet or Parsimony

13_cv_mapreduce_multivariate_BMI.py: cross-validation through the algorithm mapreduce, using ElasticNet or Parsimony.
UPD: BMI residualized

14_cv_mltva_BMI.py: cross validation using mapreduce to be sent to Gabriel

15_cv_mltva_BMI.py: cross validation of residualized BMI and images using mapreduce to be sent to Gabriel

extract_BMI_SNPs.py: effort to standardize data (bypass R extraction of original data and do it by Python)