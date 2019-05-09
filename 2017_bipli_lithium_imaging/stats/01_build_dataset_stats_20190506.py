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
import scipy
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
import re
import glob
import json

def slicedisplay(inputdir,filename,title,sliceaxis,slicenum,cmap):
    #filename = os.path.join(OUTPUT_DATA,varname+"p_vals_subj_log10.nii.gz")
    #plotting.plot_stat_map(filename, display_mode=sliceaxis, cut_coords=slicenum,
    #                  title="display_mode='"+sliceaxis+"', cut_coords="+str(slicenum))
    filename = os.path.join(inputdir,filename)
    plotting.plot_stat_map(filename, display_mode=sliceaxis, cut_coords=slicenum,
                      title=title+", display_mode='"+sliceaxis+"', cut_coords="+str(slicenum),cmap=cmap)

GENDER_MAP = {'F': 0, 'M': 1}
Lithresponse_MAP = {'Good': 1, 'Bad': 0}

#BASE_PATH = "V:/projects/BIPLi7/Clinicaldata/Analysis"
BASE_PATH = "/neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis"


#INPUT_CSV = os.path.join(BASE_PATH,"Processing","BipLipop_testnofilt.csv")
#INPUT_CSV = os.path.join(BASE_PATH,"Processing","BipLipop_minus11.csv")
INPUT_CSV = os.path.join(BASE_PATH,"BipLipop_Analysis.csv")

#INPUT_FILES_DIR = os.path.join(BASE_PATH,"Processing/Processingtestnofilter/Lithiumfiles_02_mask_b/")
#OUTPUT_DATA = os.path.join(BASE_PATH,"Processing/Processingtestnofilter/Analysisoutputs")
INPUT_FILES_DIR = os.path.join(BASE_PATH, "Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01")
#OUTPUT_DATA = os.path.join(BASE_PATH,"Processing_February_2019/Reconstruct_gridding/Processing_quantif/TPI_Analysisoutputs")
#OUTPUT_DATA = os.path.join(BASE_PATH,"Processing_February_2019/Reconstruct_gridding/Processing_quantif/TPI_Analysisoutputs_fsl")
#OUTPUT_DATA = os.path.join(BASE_PATH,"Processing_February_2019/Reconstruct_gridding/Processing_quantif/TPI_Analysisoutputs_fsl-thres3.5")
OUTPUT_DATA = os.path.join(INPUT_FILES_DIR,"stats")
WD = OUTPUT_DATA
os.makedirs(OUTPUT_DATA, exist_ok=True)


Norm_file=os.path.join(BASE_PATH,"Processing_February_2019/Reconstruct_gridding/Processing_quantif/TPI_Lithiumfiles_03_norm_b/mean_norm.nii")
os.path.isfile(Norm_file)
# Read pop csv
pop = pd.read_csv(INPUT_CSV)
pop['sex.num'] = pop["sex"].map(GENDER_MAP)
pop['Lithresp.num']=pop["lithresponse"].map(Lithresponse_MAP)

########################################################################################################################
# Select subset of participants

match_participant_id = re.compile(r"(Patient_.+).nii")
subset = [match_participant_id.findall(p)[0] for p in glob.glob(INPUT_FILES_DIR+"/Patient_*.nii")]
pop = pop[pop.path_VBM.isin(subset)]
pop.to_csv(os.path.join(WD, "population.csv"), index=False)

########################################################################################################################
# Read images
n = len(pop)
Z = np.zeros((n, 3)) # Intercept + Age + Gender
Z[:, 0] = 1 # Intercept
Y = np.zeros((n, 1)) # DX
images = list()
images_cs = list()

for i, index in enumerate(pop.index):
    cur = pop[pop.index== index]
    #print(cur)
    imagefile_name = cur.path_VBM + ".nii"
    #imagefile_path = os.path.join(INPUT_FILES_DIR,imagefile_name.as_matrix()[0])
    imagefile_path = os.path.join(INPUT_FILES_DIR,imagefile_name.values[0])
    babel_image = nibabel.load(imagefile_path)
    #images.append(babel_image.get_data().ravel())
    images.append(babel_image)
    Z[i, 1:] = np.asarray(cur[["age", "sex.num"]]).ravel()
    Y[i, 0] = cur["Lithresp.num"]

ref_img_filename = os.path.join(INPUT_FILES_DIR, pop.iloc[0].path_VBM + ".nii")
ref_img = images[0]
assert ref_img.header.get_zooms() == (2.0, 2.0, 2.0)

shape = ref_img.get_data().shape

ALL = nilearn.image.concat_imgs(images)
ALL.to_filename(os.path.join(OUTPUT_DATA, "ALL.nii.gz"))

assert np.all(pop[["age", "sex.num"]] == Z[:, 1:])


########################################################################################################################
# Compute mask
# Implicit Masking involves assuming that a lower than a givent threshold
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.

import brainomics.image_atlas

Xtot = np.vstack([im.get_data().ravel() for im in images])

brainmask_ima = nibabel.load(os.path.join(BASE_PATH, "ROIs", "Wholebrain.nii"))
brainmask_arr = brainmask_ima.get_data() != 0
assert brainmask_arr.sum() == 268822

mask_arr = ((np.mean(Xtot, axis=0) >= 0.1) & (np.std(Xtot, axis=0) >= 1e-6) & brainmask_arr.ravel()).reshape(shape)
assert mask_arr.sum() == 268822

atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=ref_img_filename,
    output=os.path.join(WD, "atlas_harvard_oxford.nii.gz"))

cereb = brainomics.image_atlas.resample_atlas_bangor_cerebellar(
    ref=ref_img_filename,
    output=os.path.join(WD, "atlas_cerebellar.nii.gz"))

mask_arr = ((atlas.get_data() + cereb.get_data()) != 0) & mask_arr
assert mask_arr.sum() == 261622

mean = np.mean(Xtot, axis=0)
mean[~mask_arr.ravel()] = 0
sd = np.std(Xtot, axis=0)
sd[~mask_arr.ravel()] = 0

zmap = np.zeros(len(mean))
zmap[sd >= 1e-6] = mean[sd >= 1e-6] / sd[sd >= 1e-6]
zmap[~mask_arr.ravel()] = 0

nibabel.Nifti1Image(mean.reshape(shape), affine=ref_img.affine).to_filename(os.path.join(OUTPUT_DATA, "mean.nii.gz"))
nibabel.Nifti1Image(zmap.reshape(shape), affine=ref_img.affine).to_filename(os.path.join(OUTPUT_DATA, "z.nii.gz"))
mask_img = nibabel.Nifti1Image(mask_arr.astype(int), affine=ref_img.affine)
mask_img.to_filename(os.path.join(OUTPUT_DATA, "mask.nii.gz"))

rois_sub_stats, rois_cort_stats, atlas_sub_img, atlas_cort_img, rois_sub_labels, rois_cort_labels =\
    brainomics.image_atlas.roi_average(maps_img=images, atlas="harvard_oxford", mask_img=mask_img)

atlas_sub_img.to_filename(os.path.join(OUTPUT_DATA, "atlas_harvard_oxford_sub.nii.gz"))
with open(os.path.join(OUTPUT_DATA, 'atlas_harvard_oxford_sub_labels.json'), 'w') as outfile:
    json.dump(rois_sub_labels, outfile)

atlas_cort_img.to_filename(os.path.join(OUTPUT_DATA, "atlas_harvard_oxford_cort.nii.gz"))
with open(os.path.join(OUTPUT_DATA, 'atlas_harvard_oxford_cort_labels.json'), 'w') as outfile:
    json.dump(rois_sub_labels, outfile)

#maps_img = images; atlas="harvard_oxford"; mask_img=mask_img

########################################################################################################################
# Center scale image by individual brain mean/std
Xbrain = Xtot[:, mask_arr.ravel()]
assert Xbrain.shape == (18, 261622)

glob_measures = pd.DataFrame(dict(
    glob_mean = Xbrain.mean(axis=1),
    glob_sd = Xbrain.std(axis=1)))

Xbrain = Xbrain - Xbrain.mean(axis=1)[:, np.newaxis]
Xbrain = Xbrain / Xbrain.std(axis=1)[:, np.newaxis]

images_cs = list()
for i in range(Xbrain.shape[0]):
    #i=0
    arr = np.zeros(shape)
    arr[mask_arr] = Xbrain[i, :]
    images_cs.append(nibabel.Nifti1Image(arr, affine=ref_img.affine))

ALLcs = nilearn.image.concat_imgs(images_cs)
ALLcs.to_filename(os.path.join(OUTPUT_DATA, "ALLcs.nii.gz"))

rois_cs_sub_stats, rois_cs_cort_stats, atlas_sub_img, atlas_cort_img, rois_sub_labels, rois_cort_labels =\
    brainomics.image_atlas.roi_average(maps_img=images_cs, atlas="harvard_oxford", mask_img=mask_img)

with pd.ExcelWriter(os.path.join(OUTPUT_DATA, "rois.xlsx")) as writer:
    pop.to_excel(writer, sheet_name='population', index=False)
    glob_measures.to_excel(writer, sheet_name='glob_measures', index=False)
    rois_cort_stats.to_excel(writer, sheet_name='rois_cort_stats', index=False)
    rois_sub_stats.to_excel(writer, sheet_name='rois_sub_stats', index=False)
    rois_cs_cort_stats.to_excel(writer, sheet_name='rois_cs_cort_stats', index=False)
    rois_cs_sub_stats.to_excel(writer, sheet_name='rois_cs_sub_stats', index=False)



########################################################################################################################
# Statistics Voxel Based

import pandas as pd
import mulm
#import seaborn as sns

def univar_stats(Y, X, path_prefix, mask_img, threspval = 10 ** -3, threstval=3, two_tailed=False):

    contrasts = [1] + [0] *(X.shape[1] - 1)
    mod = mulm.MUOLS(Y, X)
    tvals, pvals, df = mod.fit().t_test(contrasts, pval=True, two_tailed=False)

    print([[thres, np.sum(pvals <thres), np.sum(pvals <thres)/pvals.size] for thres in 10. ** np.array([-4, -3, -2])])
    # {'voxsize': 1.5, 'smoothing': 0, 'target': 'dx_num'}
    # [[0.0001, 23068, 0.058190514149063371], [0.001, 47415, 0.11960738808643315], [0.01, 96295, 0.24291033292804132]]

    tstat_arr = np.zeros(mask_arr.shape)
    pvals_arr = np.zeros(mask_arr.shape)

    pvals_arr[mask_arr] = -np.log10(pvals[0])
    tstat_arr[mask_arr] = tvals[0]

    pvals_img = nibabel.Nifti1Image(pvals_arr, affine=mask_img.affine)
    pvals_img.to_filename(path_prefix + "_vox_p_tstat-mulm_log10.nii.gz")

    tstat_img = nibabel.Nifti1Image(tstat_arr, affine=mask_img.affine)
    tstat_img.to_filename(path_prefix + "_tstat-mulm.nii.gz")

    fig = plt.figure(figsize=(13.33,  7.5 * 4))
    ax = fig.add_subplot(411)
    ax.set_title("-log pvalues >%.2f"%  -np.log10(threspval))
    plotting.plot_glass_brain(pvals_img, threshold=-np.log10(threspval), figure=fig, axes=ax)

    ax = fig.add_subplot(412)
    ax.set_title("T-stats T>%.2f" % threstval)
    plotting.plot_glass_brain(tstat_img, threshold=threstval, figure=fig, axes=ax)

    ax = fig.add_subplot(413)
    ax.set_title("-log pvalues >%.2f"% -np.log10(threspval))
    plotting.plot_stat_map(pvals_img, colorbar=True, draw_cross=False, threshold=-np.log10(threspval), figure=fig, axes=ax)

    ax = fig.add_subplot(414)
    ax.set_title("T-stats T>%.2f" % threstval)
    plotting.plot_stat_map(tstat_img, colorbar=True, draw_cross=False, threshold=threstval, figure=fig, axes=ax)
    plt.savefig(path_prefix +  "_tstat-mulm.png")

    return tstat_arr, pvals_arr

## MULM
#Xn=np.copy(X)
#Xn1 -= X.mean(axis=0)
#Xn1 /= X.std(axis=0)

Design = np.zeros((pop.shape[0], 3))
Design[:, 0] = 1
Design[:, 1:] = pop[["age", "sex.num"]]

threspval = 2 * 10 ** -3
threstval = np.abs(scipy.stats.t.ppf(threspval / 2, df=Design.shape[0]-3))

model_str = 'li~1+age+sex'
tmap, pmap = univar_stats(Y=Xbrain, X=Design, path_prefix=os.path.join(OUTPUT_DATA, model_str),
                         mask_img=mask_img, threspval =threspval, threstval=threstval)


#############################################################################
# FSL vbm(tfce)
# https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Randomise
import subprocess
prefix = os.path.join(OUTPUT_DATA, model_str)


pd.DataFrame(Design).to_csv(prefix +'_design.txt', header=None, index=None, sep=' ', mode='a')
subprocess.run(["fsl5.0-Text2Vest", prefix +'_design.txt', prefix +'_design.mat'], stdout=subprocess.PIPE)
os.remove(prefix +'_design.txt')

contrasts = np.array([[1] + [0] *(Design.shape[1] - 1),
                      [-1] + [0] *(Design.shape[1] - 1)])
np.savetxt(prefix +'_contrast.txt', contrasts, fmt='%i')
subprocess.run(["fsl5.0-Text2Vest", prefix +'_contrast.txt', prefix +'_contrast.mat'], stdout=subprocess.PIPE)
os.remove(prefix +'_contrast.txt')

#import subprocess
#cmd = ["fsl5.0-fslmerge", "-a", DATA_DIR + "/GM.nii.gz"] + pop.path.to_list()
#out = subprocess.run(cmd, stdout=subprocess.PIPE)


cmd = ["fsl5.0-randomise", '-i', os.path.join(OUTPUT_DATA, "ALLcs.nii.gz"), "-m",  os.path.join(OUTPUT_DATA, "mask.nii.gz"),
 "-o", prefix,
 '-d', prefix +'_design.mat',
 '-t', prefix +'_contrast.mat', '-T', '-n', '2000', "-C", "3"]

print(" ".join(cmd))
"""
fsl5.0-randomise -i /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/ALLcs.nii.gz -m /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/mask.nii.gz -o /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/li~1+age+sex -d /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/li~1+age+sex_design.mat -t /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/li~1+age+sex_contrast.mat -T -n 2000 -C 4

# 3
fsl5.0-randomise -i /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/ALLcs.nii.gz -m /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/mask.nii.gz -o /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/li~1+age+sex -d /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/li~1+age+sex_design.mat -t /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/li~1+age+sex_contrast.mat -T -n 2000 -C 3

# 2.5
fsl5.0-randomise -i /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/ALLcs.nii.gz -m /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/mask.nii.gz -o /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/li~1+age+sex -d /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/li~1+age+sex_design.mat -t /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/li~1+age+sex_contrast.mat -T -n 2000 -C 2.5

# 3.5
fsl5.0-randomise -i /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/ALLcs.nii.gz -m /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/mask.nii.gz -o /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/li~1+age+sex -d /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/li~1+age+sex_design.mat -t /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/li~1+age+sex_contrast.mat -T -n 2000 -C 3.5

# 3.7
fsl5.0-randomise -i /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/ALLcs.nii.gz -m /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/mask.nii.gz -o /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/li~1+age+sex -d /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/li~1+age+sex_design.mat -t /neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/Processing_February_2019/Reconstruct_gridding/Processing_quantif_minartefact/TPI_Lithiumfiles_01/stats/li~1+age+sex_contrast.mat -T -n 2000 -C 3.7

~/git/scripts/brainomics/image_clusters_analysis_nilearn.py li~1+age+sex_tstat1.nii.gz -o li~1+age+sex_tstat1 --thresh_neg_low 0 --thresh_neg_high 0 --thresh_pos_low 3 --thresh_size 10
"""


########################################################################################################################
# ROI Statistics

import statsmodels.formula.api as smfrmla

pop = pd.read_excel(os.path.join(OUTPUT_DATA, "rois.xlsx"), sheetname='population')
rois_cort_stats = pd.read_excel(os.path.join(OUTPUT_DATA, "rois.xlsx"), sheetname='rois_cort_stats')
rois_sub_stats = pd.read_excel(os.path.join(OUTPUT_DATA, "rois.xlsx"), sheetname='rois_sub_stats')

rois_cs_cort_stats = pd.read_excel(os.path.join(OUTPUT_DATA, "rois.xlsx"), sheetname='rois_cs_cort_stats')
rois_cs_sub_stats = pd.read_excel(os.path.join(OUTPUT_DATA, "rois.xlsx"), sheetname='rois_cs_sub_stats')
glob_measures = pd.read_excel(os.path.join(OUTPUT_DATA, "rois.xlsx"), sheetname='glob_measures')


def do_stat_rois(df, pop, glob_measures):
    import statsmodels.sandbox.stats.multicomp as multicomp
    import re
    df.columns = [re.sub('[^0-0a-zA-Z_]+', "", col) for col in df.columns]
    df = df[[col for col in df.columns if col.count("_mean")]]
    df.mean()

    rois = df.columns.tolist()

    #df["inter"] = 1
    df["sex"] = pop["sex"]
    df["age"] = pop["age"]
    df["glob_mean"] = glob_measures["glob_mean"]

    #df["glob_mean_roi"] = df[rois].mean(axis=1)
    #plt.plot(df["glob_mean_roi"], df["glob_mean"] , "o")
    #plt.show()

    stats_list = list()
    for roi in rois:
        #model = smfrmla.ols("%s~age+sex+glob_mean" % roi, df).fit()
        model = smfrmla.ols("%s~age+sex" % roi, df).fit()
        summary = model.summary().tables[1].data
        idx = [i for i in range(len(summary)) if summary[i][0] == 'Intercept'][0]
        res = [float(s.strip()) for s in  summary[idx][1:]]
        res[3] = model.pvalues.loc["Intercept"]
        stats_list.append([roi] + res)
        # model.params.loc["Intercept"]
        # model.tvalues.loc["Intercept"]
        # model.pvalues.loc["Intercept"]
        # model.conf_int().loc["Intercept", :]
        # model.df_resid

    stats = pd.DataFrame(stats_list,
                 columns= ["IV", "coef", "std err", "t", "pvalue", "ci0_025", "ci0_975"])
    stats["pvalue_positive"] = stats["pvalue"] / 2
    stats["pvalue_positive"][stats["t"] < 0] = 1 - stats["pvalue_positive"][stats["t"] < 0]

    _, pvals_fdr, _, _ = multicomp.multipletests(stats.pvalue_positive, alpha=0.05,
    method='fdr_bh')
    stats["pvalue_positive_fdr"] = pvals_fdr

    return(stats)


roi_sub_stats = do_stat_rois(rois_cs_cort_stats.copy(), pop, glob_measures)
roi_sub_stats = do_stat_rois(rois_cs_sub_stats.copy(), pop, glob_measures)


with pd.ExcelWriter(os.path.join(OUTPUT_DATA, "roi_stats.xlsx")) as writer:
    roi_sub_stats.to_excel(writer, sheet_name='roi_sub_stats', index=False)
    roi_cort_stats.to_excel(writer, sheet_name='roi_cort_stats', index=False)

