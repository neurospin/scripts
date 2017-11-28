# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:12:36 2016

@author: ad247405
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from brainomics import plot_utilities
import parsimony.utils.check_arrays as check_arrays
import scipy.stats


################
# Input/Output #
################

BASE_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs_3comp_patients_only"
INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs_3comp_patients_only/adni_5folds/"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,"results")
INPUT_DATASET = os.path.join(INPUT_BASE_DIR,"X.npy")
INPUT_MASK = os.path.join(BASE_DIR,"mask.npy")
INPUT_RESULTS_FILE=os.path.join(INPUT_BASE_DIR,"results.csv")
INPUT_CONFIG_FILE = os.path.join(INPUT_BASE_DIR,"config_5folds.json")
OUTPUT_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs/adni_5folds"
OUTPUT_COMPONENTS = os.path.join(OUTPUT_DIR,"components.csv")

##############
# Parameters #
##############

N_COMP = 3
INPUT_COMPONENTS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'components.npz')
INPUT_PROJECTIONS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'X_train_transform.npz')

OUTPUT_COMPONENTS_FILE_FORMAT = os.path.join(OUTPUT_DIR,
                                             '{name}.nii')

def transform(V, X, n_components, in_place=False):
    """ Project a (new) dataset onto the components.
    Return the projected data and the associated d.
    We have to recompute U and d because the argument may not have the same
    number of lines.
    The argument must have the same number of columns than the datset used
    to fit the estimator.
    """
    Xk = check_arrays(X)
    if not in_place:
        Xk = Xk.copy()
    n, p = Xk.shape
    if p != V.shape[0]:
        raise ValueError(
                    "The argument must have the same number of columns "
                    "than the datset used to fit the estimator.")
    U = np.zeros((n, n_components))
    d = np.zeros((n_components, ))
    for k in range(n_components):
        # Project on component j
        vk = V[:, k].reshape(-1, 1)
        uk = np.dot(X, vk)
        uk /= np.linalg.norm(uk)
        U[:, k] = uk[:, 0]
        dk = np.dot(uk.T, np.dot(Xk, vk))
        d[k] = dk
        # Residualize
        Xk -= dk * np.dot(uk, vk.T)
    return U, d


####################################################################

# Load data
####################################################################




X=np.load(os.path.join(BASE_DIR,'X.npy'))
y=np.load(os.path.join(INPUT_BASE_DIR,'y.npy')).reshape(X.shape[0])
label=y

X=np.load(os.path.join(BASE_DIR,'X_hallu_only.npy'))


#Define the parameter to load
params=np.array(('struct_pca', '0.1', '0.5', '0.1'))
params=np.array(('struct_pca', '0.1', '1e-06', '0.1'))
params=np.array(('sparse_pca', '0.0', '0.0', '2.0'))



components  =np.zeros((X.shape[1], 3))
fold=0 # First Fold is whole dataset
key = '_'.join([str(param)for param in params])
print "process", key
name=params[0]

# Load components and projections
####################################################################
components_filename = INPUT_COMPONENTS_FILE_FORMAT.format(fold=fold,key=key)
projections_filename = INPUT_PROJECTIONS_FILE_FORMAT.format(fold=fold,key=key)
components = np.load(components_filename)['arr_0']
projections = np.load(projections_filename)['arr_0']
assert projections.shape[1] == components.shape[1]

X=np.load(os.path.join(BASE_DIR,'X.npy'))
U_controls, d = transform(V=components, X=X[y==0], n_components=components.shape[1], in_place=False)



#Test correlation of projection with clinical DX
#####################################################################


BASE_PATH = "/neurospin/brainomics/2014_pca_struct/adni"
INPUT_CSV = os.path.join(BASE_PATH,"population.csv")
# Read pop csv
pop = pd.read_csv(INPUT_CSV)

pop["PC1"] = projections[:,0]
pop["PC2"] = projections[:,1]
pop["PC3"] = projections[:,2]
pop["y"] = pop["DX.num"]

from mulm.dataframe.descriptive_statistics import describe_df_basic
from mulm.dataframe.mulm_dataframe import MULM
from statsmodels.sandbox.stats.multicomp import multipletests
from patsy import dmatrices
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages

PCS = [1, 2, 3]
for j, pc in enumerate(PCS):
    model = 'y~PC%s+Age+Sex + Center' % (pc)
    y, X = dmatrices(model, data=pop, return_type='dataframe')
    mod = sm.OLS(y, X).fit()
    #test = mod.t_test([0, 1]+[0]*(X.shape[1]-2))
    test = mod.t_test([0,0,1,0,0])
    tval =  test.tvalue[0, 0]
    pval = test.pvalue
    print ("Correlation with PC %s : %s %s" %(pc,tval,pval))



# Plot of components according to clinical status
#####################################################################


plt.plot(projections[y==0,0],projections[y==0,1],'o',label = " controls")
plt.plot(projections[y==1,0],projections[y==1,1],'o',label = " AD converters")
plt.xlabel('component 0')
plt.ylabel('component 1')
plt.legend()


plt.plot(projections[y==0,0],projections[y==0,2],'o',label = " controls")
plt.plot(projections[y==1,0],projections[y==1,2],'o',label = " AD converters")
plt.xlabel('component 0')
plt.ylabel('component 2')
plt.legend()


plt.plot(projections[y==0,1],projections[y==0,2],'o',label = " controls")
plt.plot(projections[y==1,1],projections[y==1,2],'o',label = " AD converters")
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend()



#Clustering of PreH block
##############################################################################
from sklearn import cluster
mod = cluster.KMeans(n_clusters=2)
predict = mod.fit_predict(projections[y==1,1].reshape(133,1))


plt.plot(projections[y==1,0][predict==0],projections[y==1,2][predict==0],'o',label = 'cluster A')
plt.plot(projections[y==1,0][predict==1],projections[y==1,2][predict==1],'o',label = 'cluster B')
plt.xlabel('component 0')
plt.ylabel('component 2')
plt.legend()

plt.plot(projections[y==1,0][predict==0],projections[y==1,1][predict==0],'o',label = 'cluster A')
plt.plot(projections[y==1,0][predict==1],projections[y==1,1][predict==1],'o',label = 'cluster B')
plt.xlabel('component 0')
plt.ylabel('component 1')
plt.legend()

plt.plot(projections[y==1,1][predict==0],projections[y==1,2][predict==0],'o',label = 'cluster A')
plt.plot(projections[y==1,1][predict==1],projections[y==1,2][predict==1],'o',label = 'cluster B')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend()


##############################################################################
#Time of conversion MCI to AD, 6, 12, 18 or 24 months
BASE_PATH = "/neurospin/brainomics/2014_pca_struct/adni"
INPUT_CSV = os.path.join(BASE_PATH,"population_with_converters_time.csv")
# Read pop csv
pop = pd.read_csv(INPUT_CSV)


time = pop.time_of_conversion
X=np.load(os.path.join(BASE_DIR,'X.npy'))
y=np.load(os.path.join(INPUT_BASE_DIR,'y.npy')).reshape(X.shape[0])
y=y.reshape(360)
time[y==1]


plt.plot(projections[np.array(time==6),0],projections[np.array(time==6),2],'o',label = '6 months')
plt.plot(projections[np.array(time==12),0],projections[np.array(time==12),2],'o',label = '12 months')
plt.plot(projections[np.array(time==18),0],projections[np.array(time==18),2],'o',label = '18 months')
plt.plot(projections[np.array(time==24),0],projections[np.array(time==24),2],'o',label = '24 months')
plt.xlabel('component 0')
plt.ylabel('component 2')
plt.legend(loc = 'lower right')

plt.plot(projections[np.array(time==6),0],projections[np.array(time==6),1],'o',label = '6 months')
plt.plot(projections[np.array(time==12),0],projections[np.array(time==12),1],'o',label = '12 months')
plt.plot(projections[np.array(time==18),0],projections[np.array(time==18),1],'o',label = '18 months')
plt.plot(projections[np.array(time==24),0],projections[np.array(time==24),1],'o',label = '24 months')
plt.xlabel('component 0')
plt.ylabel('component 1')
plt.legend(loc = 'lower right')

plt.plot(projections[np.array(time==6),1],projections[np.array(time==6),2],'o',label = '6 months')
plt.plot(projections[np.array(time==12),1],projections[np.array(time==12),2],'o',label = '12 months')
plt.plot(projections[np.array(time==18),1],projections[np.array(time==18),2],'o',label = '18 months')
plt.plot(projections[np.array(time==24),1],projections[np.array(time==24),2],'o',label = '24 months')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend(loc = 'lower right')
##############################################################################

##############################################################################
plt.plot(time[np.isnan(time)==False],projections[np.array(np.isnan(time) ==False),0],'o',linewidth = 1)
plt.ylabel('component 0')
plt.xlabel('Time until conversion in months')

plt.plot(time[np.isnan(time)==False],projections[np.array(np.isnan(time) ==False),1],'o',linewidth = 1)
plt.ylabel('component 1')
plt.xlabel('Time until conversion in months')

plt.plot(time[np.isnan(time)==False],projections[np.array(np.isnan(time) ==False),2],'o',linewidth = 1)
plt.ylabel('component 2')
plt.xlabel('Time until conversion in months')
##############################################################################



from mulm.dataframe.descriptive_statistics import describe_df_basic
from mulm.dataframe.mulm_dataframe import MULM
from statsmodels.sandbox.stats.multicomp import multipletests
from patsy import dmatrices
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages

INPUT_CSV = os.path.join(BASE_PATH,"population_with_converters_time.csv")
pop = pd.read_csv(INPUT_CSV)
patients= pop[np.isnan(pop["time_of_conversion"]) == False]
patients["PC1"] = projections[:,0]
patients["PC2"] = projections[:,1]
patients["PC3"] = projections[:,2]


PCS = [1, 2, 3]
for j, pc in enumerate(PCS):
    model = 'time_of_conversion~PC%s+Age+Sex + Center' % (pc)
    y, X = dmatrices(model, data=patients, return_type='dataframe')
    mod = sm.OLS(np.array(time[np.isnan(time)==False]), X).fit()
    #test = mod.t_test([0, 1]+[0]*(X.shape[1]-2))
    test = mod.t_test([0,0,1,0,0])
    tval =  test.tvalue[0, 0]
    pval = test.pvalue
    print "Correlation with PC %s : %s %s" %(pc,tval,pval)
##############################################################################


#Plot controls score computed with patients components


X=np.load(os.path.join(BASE_DIR,'X.npy'))

#X=np.load(os.path.join(BASE_DIR,'X_hallu_only.npy'))

y=np.load(os.path.join(INPUT_BASE_DIR,'y.npy')).reshape(X.shape[0])
label=y


#Define the parameter to load
params=np.array(('struct_pca', '0.1', '0.5', '0.1'))
params=np.array(('struct_pca', '0.1', '1e-06', '0.1'))
params=np.array(('sparse_pca', '0.0', '0.0', '2.0'))


components  =np.zeros((X.shape[1], 3))
fold=0 # First Fold is whole dataset
key = '_'.join([str(param)for param in params])
print "process", key
name=params[0]

# Load components and projections
####################################################################
components_filename = INPUT_COMPONENTS_FILE_FORMAT.format(fold=fold,key=key)
projections_filename = INPUT_PROJECTIONS_FILE_FORMAT.format(fold=fold,key=key)
components = np.load(components_filename)['arr_0']
projections = np.load(projections_filename)['arr_0']
assert projections.shape[1] == components.shape[1]

U_controls, d = transform(V=components, X=X[y==0], n_components=components.shape[1], in_place=False)



##############################################################################
data_comp0 = [U_controls[:,0],projections[np.array(time==24)][:,0],projections[np.array(time==18)][:,0],projections[np.array(time==12)][:,0],projections[np.array(time==6)][:,0]]
plt.boxplot(data_comp0)
plt.xticks([1,2,3,4,5],["controls","24 months", "18 months","12 months","6 months"])
plt.ylabel("score on component 1")
plt.plot(np.random.normal(1, 0.04, size=sum(y==0)),U_controls[:,0],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(2, 0.04, size=sum(time==24)),projections[np.array(time==24)][:,0],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(3, 0.04, size=sum(time==18)),projections[np.array(time==18)][:,0],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(4, 0.04, size=sum(time==12)),projections[np.array(time==12)][:,0],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(5, 0.04, size=sum(time==6)),projections[np.array(time==6)][:,0],'o',color='b',alpha = 0.2,markersize=5)



data_comp1 = [U_controls[:,1],projections[np.array(time==24)][:,1],projections[np.array(time==18)][:,1],projections[np.array(time==12)][:,1],projections[np.array(time==6)][:,1]]
plt.boxplot(data_comp1)
plt.xticks([1,2,3,4,5],["controls","24 months", "18 months","12 months","6 months"])
plt.ylabel("score on component 2")
plt.plot(np.random.normal(1, 0.04, size=sum(y==0)),U_controls[:,1],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(2, 0.04, size=sum(time==24)),projections[np.array(time==24)][:,1],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(3, 0.04, size=sum(time==18)),projections[np.array(time==18)][:,1],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(4, 0.04, size=sum(time==12)),projections[np.array(time==12)][:,1],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(5, 0.04, size=sum(time==6)),projections[np.array(time==6)][:,1],'o',color='b',alpha = 0.2,markersize=5)



data_comp2 = [U_controls[:,2],projections[np.array(time==24)][:,2],projections[np.array(time==18)][:,2],projections[np.array(time==12)][:,2],projections[np.array(time==6)][:,2]]
plt.boxplot(data_comp2)
plt.xticks([1,2,3,4,5],["controls","24 months", "18 months","12 months","6 months"])
plt.ylabel("score on component 3")
plt.plot(np.random.normal(1, 0.04, size=sum(y==0)),U_controls[:,2],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(2, 0.04, size=sum(time==24)),projections[np.array(time==24)][:,2],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(3, 0.04, size=sum(time==18)),projections[np.array(time==18)][:,2],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(4, 0.04, size=sum(time==12)),projections[np.array(time==12)][:,2],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(5, 0.04, size=sum(time==6)),projections[np.array(time==6)][:,2],'o',color='b',alpha = 0.2,markersize=5)


scipy.stats.ttest_ind(projections[:,0],U_controls[:,0])

scipy.stats.ttest_ind(projections[:,1],U_controls[:,1])

scipy.stats.ttest_ind(projections[:,2],U_controls[:,2])

##############################################################################


data = [U_controls[:,0],projections[:,0]]
plt.boxplot(data)
plt.xticks([1,2],["controls","converters"])
plt.ylabel("score on component 1")
plt.plot(np.random.normal(1, 0.04, size=sum(y==0)),U_controls[:,0],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(2, 0.04, size=sum(y==1)),projections[:][:,0],'o',color='b',alpha = 0.2,markersize=5)


data = [U_controls[:,1],projections[:,1]]
plt.boxplot(data)
plt.xticks([1,2],["controls","converters"])
plt.ylabel("score on component 2")
plt.plot(np.random.normal(1, 0.04, size=sum(y==0)),U_controls[:,1],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(2, 0.04, size=sum(y==1)),projections[:][:,1],'o',color='b',alpha = 0.2,markersize=5)



#####################################################################################
X= np.load("/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/X.npy")
X_patients = np.load("/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected/X_patients.npy")
comp = np.load("/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected/model_selectionCV/all/all/struct_pca_0.1_0.5_0.1/components.npz")['arr_0']
projections = np.load("/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected/model_selectionCV/all/all/struct_pca_0.1_0.5_0.1/X_test_transform.npz")['arr_0']



BASE_PATH = "/neurospin/brainomics/2014_pca_struct/adni"
INPUT_CSV = os.path.join(BASE_PATH,"/neurospin/brainomics/2014_pca_struct/adni/population_with_converters_time.csv")
# Read pop csv
pop = pd.read_csv(INPUT_CSV)
pop = pop[pop["DX.num"]==1]
pop["PC1"] = projections[:,0]
pop["PC2"] = projections[:,1]
pop["PC3"] = projections[:,2]
pop["y"] = pop["DX.num"]
age = (pop["Age"]).values

INPUT_CSV = os.path.join(BASE_PATH,"/neurospin/brainomics/2014_pca_struct/adni/population.csv")
pop = pd.read_csv(INPUT_CSV)
pop = pop[pop["DX.num"]==1]
mmse = pop["MMSE Total Score.sc"].values
mmse12 = pop["MMSE Total Score.m12"].values
mmse24 = pop["MMSE Total Score.m24"].values
adas = pop["ADAS11.sc"].values


INPUT_CSV = os.path.join(BASE_PATH,"/neurospin/brainomics/2014_pca_struct/adni/population.csv")
pop = pd.read_csv(INPUT_CSV)
mmse = pop["MMSE Total Score.sc"].values
mmse12 = pop["MMSE Total Score.m12"].values
mmse24 = pop["MMSE Total Score.m24"].values
adas = pop["ADAS11.sc"].values


pearsonr(projections[:,0],adas)
pearsonr(projections[:,1],adas)
pearsonr(projections[:,2],adas)

pearsonr(projections[:,0],mmse)
pearsonr(projections[:,1],mmse)
pearsonr(projections[:,2],mmse)


pearsonr(U_all[:,0],mmse)
pearsonr(U_all[:,1],mmse)
pearsonr(U_all[:,2],mmse)

pearsonr(U_all[:,0],adas)
pearsonr(U_all[:,1],adas)
pearsonr(U_all[:,2],adas)

#pearsonr(U_patients[:,0],mmse)
#pearsonr(U_patients[:,1],mmse)
#pearsonr(U_patients[:,2],mmse)

plt.plot(mmse[y==0],U_all[y==0,0],'o',color='g',label = "controls")
plt.plot(mmse[y==1],U_all[y==1,0],'o',color='r',label = "MCI patients")
plt.ylabel("Score Comp 1")
plt.xlabel("MMSE score")
plt.title("corr: 0.28, p = 2.3e-08")
plt.legend()

plt.plot(mmse[y==0],U_all[y==0,1],'o',color='g',label = "controls")
plt.plot(mmse[y==1],U_all[y==1,1],'o',color='r',label = "MCI patients")
plt.ylabel("Score Comp 2")
plt.xlabel("MMSE score")
plt.title("corr: -0.21, p = 4.9e-05")
plt.legend()

plt.plot(mmse[y==0],U_all[y==0,2],'o',color='g',label = "controls")
plt.plot(mmse[y==1],U_all[y==1,2],'o',color='r',label = "MCI patients")
plt.ylabel("Score Comp 3")
plt.xlabel("MMSE score")
plt.title("corr: 0.28, p = 6.7e-08")
plt.legend()

plt.plot(adas[y==0],U_all[y==0,0],'o',color='g',label = "controls")
plt.plot(adas[y==1],U_all[y==1,0],'o',color='r',label = "MCI patients")
plt.ylabel("Score Comp 1")
plt.xlabel("ADAS score")
plt.title("corr: -0.34, p = 4.2e-11")
plt.legend()

plt.plot(adas[y==0],U_all[y==0,1],'o',color='g',label = "controls")
plt.plot(adas[y==1],U_all[y==1,1],'o',color='r',label = "MCI patients")
plt.ylabel("Score Comp 2")
plt.xlabel("ADAS score")
plt.title("corr: 0.26, p = 3.6e-07")
plt.legend()

plt.plot(adas[y==0],U_all[y==0,2],'o',color='g',label = "controls")
plt.plot(adas[y==1],U_all[y==1,2],'o',color='r',label = "MCI patients")
plt.ylabel("Score Comp 3")
plt.xlabel("ADAS score")
plt.title("corr: -0.35, p = 4.5e-12")
plt.legend()

U_controls, d = transform(V=comp, X=X[y==0], n_components=components.shape[1], in_place=False)
U_patients, d = transform(V=comp, X=X[y==1], n_components=components.shape[1], in_place=False)
U_all, d = transform(V=comp, X=X, n_components=components.shape[1], in_place=False)



plt.plot(adas,U_patients[:,0],'o')
plt.ylabel("Score Comp 3")
plt.xlabel("ADAS score")
plt.title("corr: -0.35, p = 4.5e-12")
pearsonr(U_patients[:,0],adas)
pearsonr(U_patients[:,1],adas)
pearsonr(U_patients[:,2],adas)










data_comp0 = [projections[np.array(time==24)][:,0],projections[np.array(time==18)][:,0],projections[np.array(time==12)][:,0],projections[np.array(time==6)][:,0]]
plt.boxplot(data_comp0)
plt.xticks([1,2,3,4],["24 months", "18 months","12 months","6 months"])
plt.ylabel("score on component 1")
plt.plot(np.random.normal(1, 0.04, size=sum(time==24)),projections[np.array(time==24)][:,0],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(2, 0.04, size=sum(time==18)),projections[np.array(time==18)][:,0],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(3, 0.04, size=sum(time==12)),projections[np.array(time==12)][:,0],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(4, 0.04, size=sum(time==6)),projections[np.array(time==6)][:,0],'o',color='b',alpha = 0.2,markersize=5)


data_comp1 = [projections[np.array(time==24)][:,1],projections[np.array(time==18)][:,0],projections[np.array(time==12)][:,0],projections[np.array(time==6)][:,0]]
plt.boxplot(data_comp1)
plt.xticks([1,2,3,4],["24 months", "18 months","12 months","6 months"])
plt.ylabel("score on component 1")
plt.plot(np.random.normal(1, 0.04, size=sum(time==24)),projections[np.array(time==24)][:,1],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(2, 0.04, size=sum(time==18)),projections[np.array(time==18)][:,1],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(3, 0.04, size=sum(time==12)),projections[np.array(time==12)][:,1],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(4, 0.04, size=sum(time==6)),projections[np.array(time==6)][:,1],'o',color='b',alpha = 0.2,markersize=5)



data_comp2 = [projections[np.array(time==24)][:,2],projections[np.array(time==18)][:,0],projections[np.array(time==12)][:,0],projections[np.array(time==6)][:,0]]
plt.boxplot(data_comp2)
plt.xticks([1,2,3,4],["24 months", "18 months","12 months","6 months"])
plt.ylabel("score on component 1")
plt.plot(np.random.normal(1, 0.04, size=sum(time==24)),projections[np.array(time==24)][:,2],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(2, 0.04, size=sum(time==18)),projections[np.array(time==18)][:,2],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(3, 0.04, size=sum(time==12)),projections[np.array(time==12)][:,2],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(4, 0.04, size=sum(time==6)),projections[np.array(time==6)][:,2],'o',color='b',alpha = 0.2,markersize=5)

data_age= [age[np.array(time==24)][:,2],age[np.array(time==18)][:,0],age[np.array(time==12)][:,0],age[np.array(time==6)][:,0]]
plt.boxplot(data_comp2)
plt.xticks([1,2,3,4],["24 months", "18 months","12 months","6 months"])
plt.ylabel("score on component 1")
plt.plot(np.random.normal(1, 0.04, size=sum(time==24)),projections[np.array(time==24)][:,2],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(2, 0.04, size=sum(time==18)),projections[np.array(time==18)][:,2],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(3, 0.04, size=sum(time==12)),projections[np.array(time==12)][:,2],'o',color='b',alpha = 0.2,markersize=5)
plt.plot(np.random.normal(4, 0.04, size=sum(time==6)),projections[np.array(time==6)][:,2],'o',color='b',alpha = 0.2,markersize=5)




PCS = [1, 2, 3]
for j, pc in enumerate(PCS):
    model = 'time_of_conversion~PC%s+Age+Sex + Center' % (pc)
    y, X = dmatrices(model, data=patients, return_type='dataframe')
    mod = sm.OLS(np.array(time[np.isnan(time)==False]), X).fit()
    #test = mod.t_test([0, 1]+[0]*(X.shape[1]-2))
    test = mod.t_test([0,0,1,0,0])
    tval =  test.tvalue[0, 0]
    pval = test.pvalue
    print ("Correlation with PC %s : %s %s" %(pc,tval,pval))