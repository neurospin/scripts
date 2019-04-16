# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:25:41 2016

@author: ad247405

Compute mask, concatenate masked non-smoothed images for all the subjects.
Build X, y, and mask

INPUT:
- subject_list.txt:
- population.csv

OUTPUT_ICAARZ:
- mask.nii
- y.npy
- X.npy = intercept + Age + Gender + Voxel
"""

import os
import numpy as np
import pandas as pd
import nibabel
#import brainomics.image_atlas
import nilearn
from nilearn import plotting
from mulm import MUOLS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats
#import array_utils
#import proj_classif_config


def slicedisplay(inputdir,filename,title,sliceaxis,slicenum,cmap):
    #filename = os.path.join(OUTPUT_DATA,varname+"p_vals_subj_log10.nii.gz")
    #plotting.plot_stat_map(filename, display_mode=sliceaxis, cut_coords=slicenum,
    #                  title="display_mode='"+sliceaxis+"', cut_coords="+str(slicenum))
    filename = os.path.join(inputdir,filename)
    plotting.plot_stat_map(filename, display_mode=sliceaxis, cut_coords=slicenum,
                      title=title+", display_mode='"+sliceaxis+"', cut_coords="+str(slicenum),cmap=cmap)    



GENDER_MAP = {'F': 0, 'M': 1}
Lithresponse_MAP = {'Good': 1, 'Bad': 0}

BASE_PATH = "V:/projects/BIPLi7/Clinicaldata/Analysis"
#BASE_PATH = "/neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis


#INPUT_CSV_ICAAR = os.path.join(BASE_PATH,"Processing","BipLipop_testnofilt.csv")
#INPUT_CSV_ICAAR = os.path.join(BASE_PATH,"Processing","BipLipop_minus11.csv")
INPUT_CSV_ICAAR = os.path.join(BASE_PATH,"BipLipop_analysis.csv")

#INPUT_FILES_DIR = os.path.join(BASE_PATH,"Processing/Processingtestnofilter/Lithiumfiles_02_mask_b/")
#OUTPUT_DATA = os.path.join(BASE_PATH,"Processing/Processingtestnofilter/Analysisoutputs")
INPUT_FILES_DIR = os.path.join(BASE_PATH,"Processing_February_2019/Reconstruct_gridding/Processing_quantif/TPI_Lithiumfiles_02_mask_b")
OUTPUT_DATA = os.path.join(BASE_PATH,"Processing_February_2019/Reconstruct_gridding/Processing_quantif/TPI_Analysisoutputs")
Norm_file=os.path.join(BASE_PATH,"Processing_February_2019/Reconstruct_gridding/Processing_quantif/TPI_Lithiumfiles_03_norm_b/mean_norm.nii")
os.path.isfile(Norm_file)
# Read pop csv
pop = pd.read_csv(INPUT_CSV_ICAAR)
pop['sex.num'] = pop["sex"].map(GENDER_MAP)
pop['Lithresp.num']=pop["lithresponse"].map(Lithresponse_MAP)
#############################################################################
# Read images
n = len(pop)
Z = np.zeros((n, 3)) # Intercept + Age + Gender
Z[:, 0] = 1 # Intercept
Y = np.zeros((n, 1)) # DX
images = list()
for i, index in enumerate(pop.index):
    cur = pop[pop.index== index]
    #print(cur)
    imagefile_name = cur.path_VBM + ".nii"
    #imagefile_path = os.path.join(INPUT_FILES_DIR,imagefile_name.as_matrix()[0])
    imagefile_path = os.path.join(INPUT_FILES_DIR,imagefile_name.values[0])
    babel_image = nibabel.load(imagefile_path)
    images.append(babel_image.get_data().ravel())
    Z[i, 1:] = np.asarray(cur[["age", "sex.num"]]).ravel()
    Y[i, 0] = cur["Lithresp.num"]

shape = babel_image.get_data().shape

#############################################################################
# Compute mask
# Implicit Masking involves assuming that a lower than a givent threshold
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.
Xtot = np.vstack(images)

mask_ima = nibabel.load(os.path.join(BASE_PATH, "ROIs", "Wholebrain.nii"))
mask_arr = mask_ima.get_data() != 0

#############################################################################

# Save data X and y
X = Xtot[:, mask_arr.ravel()]
#Use mean imputation, we could have used median for age
#Remove nan lines 
X = np.nan_to_num(X)
Xallmean=np.mean(X)

np.save(os.path.join(OUTPUT_DATA, "X.npy"), X)
np.save(os.path.join(OUTPUT_DATA, "Z.npy"), Z)
np.save(os.path.join(OUTPUT_DATA, "Y.npy"), Y)


###############################################################################
#############################################################################
import pandas as pd
#import seaborn as sns

X = np.load(os.path.join(OUTPUT_DATA, "X.npy"))
Z = np.load(os.path.join(OUTPUT_DATA, "Z.npy"))
#X=X[0:9,:]
#X=X*1000
X = X - X.mean(axis=1)[:, np.newaxis]
X = X / X.std(axis=1)[:, np.newaxis]

#Xn=np.copy(X)
#Xn1 -= X.mean(axis=0)
#Xn1 /= X.std(axis=0)

DesignMat=Z

muols = MUOLS(Y=X,X=DesignMat)
muols.fit()
print('launching the t_test')
tvals, pvals_ttest, df1 = muols.t_test(contrasts=[1, 0, 0], pval=True)
#%time tvals, pvals_ttest, df1 = muols.t_test(contrasts=[1, 0, 0], pval=True)
print('t_test done')
print('launching maxT test')
#tvals2, pvals_maxT, df2 = muols.t_test_maxT(contrasts=np.array([1, 0, 0]), nperms=1000, two_tailed=False)
#print('maxT test done')
tvals, pvals_maxT, df3 = muols.t_test_maxT(contrasts=np.array([1, 0, 0]), nperms=1000, two_tailed=True)
#♣tvals3, minP, df3 = muols.t_test_minP(contrasts=np.array([1, 0, 0]), nperms=5, two_tailed=True)
mhist, bins, patches= plt.hist([pvals_ttest[0,:],pvals_maxT[0,:]],
                           color=['blue','red'],
                           label=['pvals_ttest','pvals_maxT'])

mycoefs=np.zeros(mask_arr.shape)
mycoefs[mask_arr]=muols.coef[0,:]

##Wilcoxon Test
# Buisness Volume time 0
bv0 = np.random.normal(loc=3, scale=.1, size=n)
# Buisness Volume time 1
bv1 = bv0 + 0.1 + np.random.normal(loc=0, scale=.1, size=n)
# create an outlier
bv1[0] -= 10
# Paired t-test
print(stats.ttest_rel(bv0, bv1))
# Wilcoxon
print(stats.wilcoxon(bv0, bv1))



#np.correlation
#correlation test scipy test

#import scipy.stats as stats
#import matplotlib.pyplot as plt
#tvals, pvals_ttest = np.full(n_features, np.NAN), np.full(n_features, np.NAN)
#for j in range(n_features):
#    tvals[j], pvals_ttest[j] = stats.ttest_ind(Y[grp=="g1", j], Y[grp=="g2", j],
#    equal_var=True)

#import statsmodels.sandbox.stats.multicomp as multicomp
#_, pvals_ttest_fwer, _, _ = multicomp.multipletests(pvals_ttest, alpha=0.05,
#method='bonferroni')
#n_features=np.size(X,1)
#n_info = int(n_features/10)
#TP = np.sum(pvals_ttest_fwer[:n_info ] < 0.05) # True Positives
#FP = np.sum(pvals_ttest_fwer[n_info: ] < 0.05) # False Positives
#print("FWER correction, FP: %i, TP: %i" % (FP, TP))

pvallog_ttest=-np.log10(pvals_ttest[0])
pvallog_maxT=-np.log10(pvals_maxT[0])

pd.Series(tvals.ravel()).describe()
pd.Series(pvals_ttest.ravel()).describe()
pd.Series(pvallog_ttest.ravel()).describe()
pd.Series(pvals_maxT.ravel()).describe()
pd.Series(pvallog_maxT.ravel()).describe()
#check for multiple comparison, Bonferonni and/or False Discovery Rate

#arr = np.zeros(mask_arr.shape); arr[mask_arr] = (Xmeannorm)
#out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())

save=True
display=True
sliceaxis="allsave"
slicenum=10
slicenum=[-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
predict=False

varname=''
varname='stdchange_'

if predict:
    
    Xtestf = np.load(os.path.join(OUTPUT_DATA, "X.npy"))
    Xtest=Xtestf[9,3:]
    Ztest=Xtestf[9,0:3]
    #Xtest -= Xtest.mean(axis=0)
    #Xtest /= Xtest.std(axis=0)
    yvals=muols.predict(Ztest)
    yvalsn= (yvals*Xtest.std(axis=0)+Xtest.mean(axis=0))
    yvalsarr = np.zeros(mask_arr.shape);
    yvalsarr[mask_arr] = yvalsn
    out_im = nibabel.Nifti1Image(yvalsarr, affine=mask_ima.get_affine())
    out_im.to_filename(os.path.join(OUTPUT_DATA,"test_im10.nii.gz"))

if save:
    
    #Saving the log10 of pvals of the ttest
    arrlogp_tt = np.zeros(mask_arr.shape); arrlogp_tt[mask_arr] = pvallog_ttest
    out_imlogp = nibabel.Nifti1Image(arrlogp_tt, affine=mask_ima.get_affine())
    out_imlogp.to_filename(os.path.join(OUTPUT_DATA,varname+"p_vals_ttest_log10.nii.gz"))
    
    #Saving the log10 of pvals of the MaxT test
    arrlogp_tm = np.zeros(mask_arr.shape); arrlogp_tm[mask_arr] = pvallog_maxT
    out_imlogp = nibabel.Nifti1Image(arrlogp_tm, affine=mask_ima.get_affine())
    out_imlogp.to_filename(os.path.join(OUTPUT_DATA,varname+"p_vals_maxT_log10.nii.gz"))

    #Saving the thresholded values of the log10 of pvals of the ttest
    pvallog_ttest_threshold = np.zeros(mask_arr.shape);
    #pvallog_ttest_threshold[mask_arr]=pvallog_ttest>3
    pvallog_ttest_threshold[mask_arr]=pvallog_ttest>1
    out_impval = nibabel.Nifti1Image(pvallog_ttest_threshold, affine=mask_ima.get_affine())
    out_impval.to_filename(os.path.join(OUTPUT_DATA,varname+"p_vals_ttest_threshold_log10.nii.gz"))    

    #Saving the thresholded values of the log10 of pvals of the maxT test
    pvallog_maxT_threshold=np.zeros(mask_arr.shape);
    #pvallog_maxT_threshold[mask_arr]=pvallog_maxT>3;
    pvallog_maxT_threshold[mask_arr]=pvallog_maxT>1;
    out_impval = nibabel.Nifti1Image(pvallog_maxT_threshold, affine=mask_ima.get_affine())
    out_impval.to_filename(os.path.join(OUTPUT_DATA,varname+"p_vals_maxT_threshold_log10.nii.gz"))        
    
    #Saving the pvals of the ttest
    arrpval = np.zeros(mask_arr.shape); arrpval[mask_arr] = (pvals_ttest[0])
    out_impval = nibabel.Nifti1Image(arrpval, affine=mask_ima.get_affine())
    out_impval.to_filename(os.path.join(OUTPUT_DATA,varname+"p_vals_ttest.nii.gz"))
    
    #Saving the pvals of the maxT test
    arrpval = np.zeros(mask_arr.shape); arrpval[mask_arr] = (pvals_maxT[0])
    out_impval = nibabel.Nifti1Image(arrpval, affine=mask_ima.get_affine())
    out_impval.to_filename(os.path.join(OUTPUT_DATA,varname+"p_vals_maxT.nii.gz"))    
    
    #Saving the tvals of the MaxT test
    arrtval = np.zeros(mask_arr.shape); arrtval[mask_arr] = tvals[0]
    out_imtval = nibabel.Nifti1Image(arrtval, affine=mask_ima.get_affine())
    out_imtval.to_filename(os.path.join(OUTPUT_DATA,varname+"t_vals.nii.gz"))
    
    #Saving the thresholded tvals of the MaxT test
    arrtval_threshold= arrtval*pvallog_maxT_threshold
    out_imtval = nibabel.Nifti1Image(arrtval_threshold, affine=mask_ima.get_affine())
    out_imtval.to_filename(os.path.join(OUTPUT_DATA,varname+"t_vals_threshold.nii.gz"))
    
    #saving coefficients
    out_imtval = nibabel.Nifti1Image(mycoefs, affine=mask_ima.get_affine())
    out_imtval.to_filename(os.path.join(OUTPUT_DATA,varname+"coef_vals.nii.gz"))
    
    #saving thresholded coefficients
    coefvalthreshold=mycoefs*pvallog_maxT_threshold
    out_imtval = nibabel.Nifti1Image(coefvalthreshold, affine=mask_ima.get_affine())
    out_imtval.to_filename(os.path.join(OUTPUT_DATA,varname+"coef_vals_threshold.nii.gz"))
    
    arrcoefmeanval = np.zeros(mask_arr.shape); arrcoefmeanval = mycoefs/np.mean(X) #arrcoefmeanval = mycoefs/Xallmean
    out_imtval = nibabel.Nifti1Image(arrcoefmeanval, affine=mask_ima.get_affine())
    out_imtval.to_filename(os.path.join(OUTPUT_DATA,varname+"coef_vals_mean.nii.gz"))
    
    coefvalmeanthreshold= arrcoefmeanval*pvallog_maxT_threshold
    out_imtval = nibabel.Nifti1Image(coefvalmeanthreshold, affine=mask_ima.get_affine())
    out_imtval.to_filename(os.path.join(OUTPUT_DATA,varname+"coef_vals_threshold_mean.nii.gz"))    
    
    if os.path.isfile(Norm_file):
        norm_vals=nibabel.load(Norm_file)
        highnorm = np.zeros(mask_arr.shape);
        highnorm=norm_vals.get_data()>1.2
        highnorm_highpval=highnorm*pvallog_ttest_threshold
        normvals_highpval=norm_vals.get_data()*pvallog_ttest_threshold

        out_imtval = nibabel.Nifti1Image(highnorm_highpval, affine=mask_ima.get_affine())
        out_imtval.to_filename(os.path.join(OUTPUT_DATA,varname+"highnorm_highpvals.nii.gz"))  

        out_imtval = nibabel.Nifti1Image(normvals_highpval, affine=mask_ima.get_affine())       
        out_imtval.to_filename(os.path.join(OUTPUT_DATA,varname+"normvals_highpvals.nii.gz"))      
        
    
if display:
    N = 21
    import scipy.stats
    threspval = 2 * 10 ** -3
    threstval = np.abs(scipy.stats.t.ppf(threspval / 2, df=N-3)) 
    filename = os.path.join(OUTPUT_DATA,varname+"p_vals_ttest_log10.nii.gz")
    nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "pvals (-log10) map",cmap=plt.cm.bwr,threshold=-np.log10(threspval))
        
    filename = os.path.join(OUTPUT_DATA,varname+"t_vals.nii.gz")
    nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic map",threshold=threstval)
    ##################################################################################
    #slicedisplay(OUTPUT_DATA,varname+"p_vals_subj_log10threshold.nii.gz","T-statistic pvals_ttest significant map", sliceaxis,slicenum)
    #filename = os.path.join(OUTPUT_DATA,varname+"t_vals_threshold.nii.gz")
    #nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic pvals_ttest map significant map")    
    
if sliceaxis=='x' or sliceaxis=='y' or sliceaxis=='z':
    
    slicedisplay(OUTPUT_DATA,varname+"p_vals_maxT_log10.nii.gz","T-statistic pvals_ttest log10 map",sliceaxis,slicenum,'cold_hot')
    #slicedisplay(OUTPUT_DATA,varname+"p_vals_base.nii.gz","T-statistic pvals_ttest map",sliceaxis,slicenum,'cold_hot')
    slicedisplay(OUTPUT_DATA,varname+"t_vals_threshold.nii.gz","T-statistic Tvals map", sliceaxis,slicenum,'bwr')
    slicedisplay(OUTPUT_DATA,varname+"p_vals_maxT_threshold_log10.nii.gz","T-statistic pvals_ttest significant map", sliceaxis,slicenum,'Oranges')
    slicedisplay(OUTPUT_DATA,varname+"coef_vals.nii.gz","T-statistic coef map", sliceaxis,slicenum,'bwr')
    slicedisplay(OUTPUT_DATA,varname+"coef_vals_threshold.nii.gz","T-statistic coef significant map", sliceaxis,slicenum,'cold_hot')
    slicedisplay(OUTPUT_DATA,varname+"coef_vals_threshold_mean.nii.gz","T-statistic coef map", sliceaxis,slicenum,'bwr')

elif sliceaxis=='all':
    for sliceaxis in ['x','y','z']:
        slicedisplay(OUTPUT_DATA,varname+"p_vals_maxT_log10.nii.gz","T-statistic pvals_ttest log10 map",sliceaxis,slicenum,'cold_hot')
        slicedisplay(OUTPUT_DATA,varname+"p_vals_maxT_threshold_log10.nii.gz","T-statistic pvals_ttest map",sliceaxis,slicenum,'cold_hot')
        slicedisplay(OUTPUT_DATA,varname+"t_vals_threshold.nii.gz","T-statistic Tvals map", sliceaxis,slicenum,'bwr')
        #slicedisplay(OUTPUT_DATA,varname+"p_vals_base.nii.gz","T-statistic pvals_ttest significant map", sliceaxis,slicenum,'Oranges')
        slicedisplay(OUTPUT_DATA,varname+"coef_vals.nii.gz","T-statistic coef map", sliceaxis,slicenum,'bwr')
        slicedisplay(OUTPUT_DATA,varname+"coef_vals_threshold.nii.gz","T-statistic coef threshold map", sliceaxis,slicenum,'bwr')
        slicedisplay(OUTPUT_DATA,varname+"coef_vals_mean.nii.gz","T-statistic coef mean map", sliceaxis,slicenum,'bwr')
        slicedisplay(OUTPUT_DATA,varname+"coef_vals_threshold_mean.nii.gz","T-statistic coef mean significant map", sliceaxis,slicenum,'bwr')
        
elif sliceaxis=='allsave':
        
    pdffilepath=os.path.join(OUTPUT_DATA,varname+"alldata.pdf")
    pdf = PdfPages(pdffilepath)
    for sliceaxis in ['x','y','z']:

        slicedisplay(OUTPUT_DATA,varname+"p_vals_maxT_log10.nii.gz","T-statistic pvals_ttest log10 map",sliceaxis,slicenum,'cold_hot')
        plt.savefig(os.path.join(OUTPUT_DATA,varname+"pvals_ttest_log10_"+sliceaxis+".pdf"))
        pdf.savefig()
        #slicedisplay(OUTPUT_DATA,varname+"p_vals_threshold_log10.nii.gz","T-statistic pvals_ttest map",sliceaxis,slicenum,'cold_hot')
        #plt.savefig(os.path.join(OUTPUT_DATA,"pvals_ttest_threshold_log10_"+sliceaxis+".pdf"))
        #pdf.savefig()
        slicedisplay(OUTPUT_DATA,varname+"t_vals_threshold.nii.gz","T-statistic Tvals map", sliceaxis,slicenum,'bwr')
        plt.savefig(os.path.join(OUTPUT_DATA,varname+"tvals_"+sliceaxis+".pdf"))
        pdf.savefig()
        #slicedisplay(OUTPUT_DATA,varname+"p_vals_base.nii.gz","T-statistic pvals_ttest significant map", sliceaxis,slicenum,'Oranges')
        slicedisplay(OUTPUT_DATA,varname+"coef_vals.nii.gz","T-statistic coef map", sliceaxis,slicenum,'bwr')
        plt.savefig(os.path.join(OUTPUT_DATA,varname+"coef_vals_"+sliceaxis+".pdf"))
        pdf.savefig()
        slicedisplay(OUTPUT_DATA,varname+"coef_vals_threshold.nii.gz","T-statistic coef significant map", sliceaxis,slicenum,'bwr')
        plt.savefig(os.path.join(OUTPUT_DATA,varname+"coef_vals_threshold_"+sliceaxis+".pdf"))
        pdf.savefig()
        slicedisplay(OUTPUT_DATA,varname+"coef_vals_mean.nii.gz","T-statistic coef mean map", sliceaxis,slicenum,'bwr')
        plt.savefig(os.path.join(OUTPUT_DATA,varname+"coef_vals_mean_"+sliceaxis+".pdf"))
        pdf.savefig()
        #slicedisplay(OUTPUT_DATA,varname+"coef_vals_threshold_mean.nii.gz","T-statistic coef mean significant map", sliceaxis,slicenum,'bwr')
        #plt.savefig(os.path.join(OUTPUT_DATA,"coef_vals_threshold_mean_"+sliceaxis+".pdf"))
        #pdf.savefig()
        
    pdf.close()