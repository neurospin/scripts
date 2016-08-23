"""import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt

comp0 = np.random.randn(1000) - 5 # samples of the 1st component
comp1 = np.random.randn(1000) + 5 # samples of the 2nd component

x = np.hstack((comp0, comp1)).reshape(-1, 1) # merge them

gmm = mixture.GMM(n_components=2) # gmm for two components
gmm.fit(x) # train it!

linspace = np.linspace(-10, 10, 1000).reshape(-1, 1)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.hist(x, 100) # draw samples
ax2.plot(linspace, np.exp(gmm.score_samples(linspace)[0]), 'r') # draw GMM
plt.show()


python -m brainvisa.axon.runprocess whitemeshdepthmap /neurospin/tmp/yann/HCP/databaseBV/hcp/100206/t1mri/BL/default_analysis/segmentation/mesh/100206_Lwhite.gii /neurospin/tmp/yann/HCP/databaseBV/hcp/100206/t1mri/BL/default_analysis/segmentation/Lcortex_100206.nii.gz /neurospin/tmp/yann/HCP/databaseBV/hcp/100206/t1mri/BL/default_analysis/segmentation/100206_Lwhite_depth.gii 10.0
"""

import os, glob, re, json, time, shutil


path0 = '/media/yl247234/SAMSUNG/hcp/databaseBV/hcp/'
temp_file_s_ids='/home/yl247234/s_ids_lists/s_ids.json'
sides = ['R', 'L']
with open(temp_file_s_ids, 'r') as f:
    data = json.load(f)
s_ids  = list(json.loads(data))

for j,s_id in enumerate(s_ids):
    folder = os.path.join(path0, s_id, "t1mri/BL/default_analysis_sym")
    if os.path.exists(folder):
        print folder
        shutil.rmtree(folder)

for side in sides:
    for j,s_id in enumerate(s_ids):
        filename =  os.path.join(path0, s_id, "t1mri/BL/default_analysis/segmentation/",
                                 s_id+"_"+side+"white_depth_on_atlas.gii")
        filename_dest = os.path.join(path0, s_id, "t1mri", "BL",
                                     "default_analysis", "segmentation",
                                     "mesh", "surface_analysis_sym", s_id+"_"+side+"white_depth_on_atlas.gii")
        filename_org = os.path.join(path0, s_id, "t1mri", "BL",
                                     "default_analysis", "segmentation",
                                     "mesh", "surface_analysis", s_id+"_"+side+"white_depth_on_atlas.gii")
        if os.path.isfile(filename_dest):
            print filename_dest
            shutil.move(filename_dest, filename_org)
            shutil.move(filename_dest+".minf", filename_org+".minf")
