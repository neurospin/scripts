# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:14:06 2015

@author: fh235918
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:33:51 2015

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import pandas
import pickle
GCTAOUT = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/'

# read imputed imagen data
#############################################
isnps = pickle.load(open('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/height_imputed_snps.pickle'))
maf = np.sum(isnps.data,axis = 0)/(2.*isnps.data.shape[0])
datas = {'snps':isnps.get_meta()[0].tolist(),
         'maf': maf}
imagenImputed = pandas.DataFrame(datas, columns=['snps', 'maf'],index=isnps.get_meta()[0].tolist())


fin = ('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/'
       'imagen_subcortCov_NP.csv')
covar = pandas.read_csv(fin, sep=' ', dtype={0:str, 1:str})
covar.index = covar['IID']
covar = covar[['FID','IID','Age','ScanningCentre', 'Sex']]

# get the information from the plosOne paper.
# comfront with IMAGEN imputed data
#############################################
plosList = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/SNPheight.csv'
orig = pandas.DataFrame.from_csv(plosList, sep=';')
orig['Beta'] = [float(i.replace('?','-')) for i in orig['Beta']]
orig['Freq'] = [float(i) for i in orig['Freq']]

plosOne = pandas.merge(orig[['A1', 'A2','Freq', 'Beta']], imagenImputed,
                       left_index=True, right_index=True, how='inner')
plosOne = plosOne.join(pandas.Series(plosOne['Freq']- plosOne['maf'],name='Diff'))
# reorder
plosOne = plosOne.loc[ isnps.get_meta()[0].tolist()]

# read height
#############################################
fname = GCTAOUT + '/height.phe'
height = pandas.read_csv(fname, sep='\t', dtype={1:str,0:str},header=None)
height.columns = ['FID','IID','height']
height.index = height['IID']



# create the PgS : 
beta  = np.asarray(plosOne['Beta']).reshape(173,-1)
PgS = np.dot(isnps.data, beta).reshape(-1)
studyPgS  = pandas.DataFrame({'PgS':PgS, 'IID':isnps.iid.reshape(-1)})
studyPgS = pandas.merge(pandas.merge(covar, height, how='inner',on='IID'), 
                        studyPgS, on='IID')
studyPgS = studyPgS[[u'FID_x', u'IID', u'Age', u'ScanningCentre', u'Sex', u'height', u'PgS']]
studyPgS.columns = [u'FID', u'IID', u'Age', u'ScanningCentre', u'Sex', u'height', u'PgS']
#studyPgS[u'Sex'].replace({0:'Male', 1:'Female'}, inplace=True)
#studyPgS[u'ScanningCentre'].replace({1:'Centre_1', 2:'Centre_2', 3:'Centre_3', 4:'Centre_4',
#     5:'Centre_5', 6:'Centre_6', 7:'Centre_7', 8:'Centre_8'}, inplace=True)

covariate = np.matrix(pandas.get_dummies(studyPgS['ScanningCentre'],
                                              prefix='Centre')[range(7)])
Cov = np.hstack((covariate, np.asarray(studyPgS[['Sex', 'Age']])))
X = np.asarray(studyPgS['PgS'])
y = studyPgS['height']
X.reshape(-1,1)
X_ = np.matrix(X).T
from sklearn.linear_model import LinearRegression
Y = y - LinearRegression().fit(Cov,y).predict(Cov)
lm = LinearRegression()
lm.fit(X_,Y)
print 'lm.score(X_,Y)', lm.score(X_,Y)


# get the 1596 subject used above in  order to be fair...
# get SNP and get Y
studyPgS.index = studyPgS['IID']
mask = [i in studyPgS['IID'].tolist() for i in isnps.iid]
SNP = isnps.data[np.asarray(mask),:]
permuter =  isnps.iid[np.asarray(mask)].tolist()
y = height.loc[permuter]['height']
covariate = np.matrix(pandas.get_dummies(studyPgS.loc[permuter]['ScanningCentre'],
                                              prefix='Centre')[range(7)])
Cov = np.hstack((covariate, np.asarray(studyPgS.loc[permuter][['Sex', 'Age']])))
Y = y - LinearRegression().fit(Cov,y).predict(Cov)
#FOUAD je me suis arreté la!!!!!!!!!!!!!!!
X = SNP
p = X.shape[1]
from scipy.stats import pearsonr
p_vect=np.array([])
cor_vect=np.array([])
for i in range(X.shape[1]):
    r_row, p_value = pearsonr(X[:, i],  Y)
    p_vect = np.hstack((p_vect, p_value))
    cor_vect = np.hstack((cor_vect, r_row))    
indices = np.where(p_vect <= 0.05)
print 'numbers of significant p values', len(indices[0]), 'over ', p


import p_value_correction as p_c
p_corrected = p_c.fdr(p_vect)
indices_c= np.where(p_corrected <= 0.05)
print 's0 : numbers of significant corrected p values', len(indices_c[0]), 'over ', p



import matplotlib.pyplot as plt
plt.figure(2)
plt.subplot(211)
plt.hist(p_corrected,40)
plt.title('corrected p values ')
plt.subplot(212)
plt.hist(p_vect,40)
plt.title('uncorrected p values ')
plt.show() 



import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(211)
plt.plot(p_corrected)
plt.title('corrected p values ')
plt.subplot(212)
plt.plot(p_vect)
plt.title('uncorrected p values ')
plt.show() 

plt.plot(cor_vect)
plt.show()

mask = np.where(p_corrected <= 0.05)

dif = cor_vect - beta.reshape(-1)
dif_nor = (dif -dif.mean())/dif.std()
plt.hist(dif_nor)
plt.show()

len(np.where(np.abs(dif_nor) > 1.96)[0])


plt.figure(2)
plt.plot(dif)
plt.show()





from sklearn import cross_validation
from sklearn.linear_model import  ElasticNetCV, ElasticNet
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

anova_filter = SelectKBest(score_func=f_regression, k=100)
#svr = svm.SVR(kernel="linear")
svr = svm.SVR(kernel="rbf")
anova_svr = Pipeline([('anova', anova_filter), ('svr', svr)])
cv_res = cross_validation.cross_val_score(anova_svr, X, Y, cv=10)
np.mean(cv_res)




parameters = {'svr__C': (.001, .01, .1, 1., 10., 100)}
anova_svrcv = GridSearchCV(anova_svr, parameters, n_jobs=-1, verbose=1)
print cross_validation.cross_val_score(anova_svrcv, X, y, cv=10)
#[-0.00113985 -0.00789315 -0.00962538 -0.00940644 -0.01980303]

a = [-0.01383361, 0.00518606, 0.04310046, 0.00566481, -0.05087966, 0.02595137,
  0.06297484, 0.00726542, 0.0512848, -0.0375388]

anova_filter = SelectKBest(score_func=f_regression, k=X_.shape[1])
#svr = svm.SVR(kernel="linear")
svr = svm.SVR()
anova_svr = Pipeline([('anova', anova_filter), ('svr', svr)])

parameters = {'svr__C': (.001, .01, .1, 1., 10., 100), 
              'svr__kernel': ("linear", "rbf"),
              'anova__k' : [7,15,30,45,70,90,110]}
              #'anova__k' : [100]}
anova_svrcv = GridSearchCV(anova_svr, parameters, n_jobs=-1, verbose=1)
res_cv =  cross_validation.cross_val_score(anova_svrcv, X, y, cv=5)
#array([ 0.01275864,  0.01481352, -0.00011443,  0.03102828,  0.02435049])
#0.016567301178355476



from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
print cross_validation.cross_val_score(rf, X, Y, cv=5)



#######################
#here wi test using the hyppcampus volume and the height_PGS 
################################
import pandas
import pickle
GCTAOUT = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/'

# read imputed imagen data
#############################################
isnps = pickle.load(open('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/height_imputed_snps.pickle'))
maf = np.sum(isnps.data,axis = 0)/(2.*isnps.data.shape[0])
datas = {'snps':isnps.get_meta()[0].tolist(),
         'maf': maf}
imagenImputed = pandas.DataFrame(datas, columns=['snps', 'maf'],index=isnps.get_meta()[0].tolist())


fin = ('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/'
       'imagen_subcortCov_NP.csv')
covar = pandas.read_csv(fin, sep=' ', dtype={0:str, 1:str})
covar.index = covar['IID']
covar = covar[['FID','IID','Age','ScanningCentre', 'Sex', 'Mhippo']]

# get the information from the plosOne paper.
# comfront with IMAGEN imputed data
#############################################
plosList = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/SNPheight.csv'
orig = pandas.DataFrame.from_csv(plosList, sep=';')
orig['Beta'] = [float(i.replace('?','-')) for i in orig['Beta']]
orig['Freq'] = [float(i) for i in orig['Freq']]

plosOne = pandas.merge(orig[['A1', 'A2','Freq', 'Beta']], imagenImputed,
                       left_index=True, right_index=True, how='inner')
plosOne = plosOne.join(pandas.Series(plosOne['Freq']- plosOne['maf'],name='Diff'))
# reorder
plosOne = plosOne.loc[ isnps.get_meta()[0].tolist()]

# create the PgS : 
beta  = np.asarray(plosOne['Beta']).reshape(173,-1)
PgS = np.dot(isnps.data, beta).reshape(-1)
studyPgS  = pandas.DataFrame({'PgS':PgS, 'IID':isnps.iid.reshape(-1)})
studyPgS.index = studyPgS['IID']
studyPgS = pandas.merge(covar, studyPgS, how='inner',on='IID')
studyPgS.index = studyPgS['IID']
#studyPgS = studyPgS[[u'FID_x', u'IID', u'Age', u'ScanningCentre', u'Sex', u'height', u'PgS']]
#studyPgS.columns = [u'FID', u'IID', u'Age', u'ScanningCentre', u'Sex', u'height', u'PgS']


covariate = np.matrix(pandas.get_dummies(studyPgS['ScanningCentre'],
                                              prefix='Centre')[range(7)])
Cov = np.hstack((covariate, np.asarray(studyPgS[['Sex', 'Age']])))
X = np.asarray(studyPgS['PgS'])
y = studyPgS['Mhippo']
X.reshape(-1,1)
X_ = np.matrix(X).T
from sklearn.linear_model import LinearRegression
Y = y - LinearRegression().fit(Cov,y).predict(Cov)
lm = LinearRegression()
lm.fit(X_,Y)
print 'lm.score(X_,Y)', lm.score(X_,Y)

##########################################"
#multivariate tests

from sklearn import cross_validation
from sklearn.linear_model import  ElasticNetCV, ElasticNet
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


mask = [i in studyPgS['IID'].tolist() for i in isnps.iid]
SNP = isnps.data[np.asarray(mask),:]
X = SNP
permuter =  isnps.iid[np.asarray(mask)].tolist()
y = studyPgS.loc[permuter]['Mhippo']
covariate = np.matrix(pandas.get_dummies(studyPgS.loc[permuter]['ScanningCentre'],
                                              prefix='Centre')[range(7)])
Cov = np.hstack((covariate, np.asarray(studyPgS.loc[permuter][['Sex', 'Age']])))
Y = y - LinearRegression().fit(Cov,y).predict(Cov)
anova_filter = SelectKBest(score_func=f_regression, k=100)
#svr = svm.SVR(kernel="linear")
svr = svm.SVR(kernel="rbf")
anova_svr = Pipeline([('anova', anova_filter), ('svr', svr)])
cv_res = cross_validation.cross_val_score(anova_svr, X, Y, cv=10)
np.mean(cv_res)




parameters = {'svr__C': (.001, .01, .1, 1., 10., 100)}
anova_svrcv = GridSearchCV(anova_svr, parameters, n_jobs=-1, verbose=1)
print cross_validation.cross_val_score(anova_svrcv, X, y, cv=10)

anova_filter = SelectKBest(score_func=f_regression, k=X_.shape[1])
#svr = svm.SVR(kernel="linear")
svr = svm.SVR()
anova_svr = Pipeline([('anova', anova_filter), ('svr', svr)])

parameters = {'svr__C': (.001, .01, .1, 1., 10., 100), 
              'svr__kernel': ("linear", "rbf"),
              'anova__k' : [7,15,30,45,70,90,110]}
              #'anova__k' : [100]}
anova_svrcv = GridSearchCV(anova_svr, parameters, n_jobs=-1, verbose=1)
res_cv =  cross_validation.cross_val_score(anova_svrcv, X, y, cv=5)

##############################################
#fname = GCTAOUT + '/studyPgS.csv'
#studyPgS.to_csv(fname, sep='\t', index=False, header=True,
#                 columns=[u'FID', u'IID', u'Age', u'ScanningCentre', u'Sex', u'height', u'PgS'])
