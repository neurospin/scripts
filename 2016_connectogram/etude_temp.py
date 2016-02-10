
import os, glob, re
import pandas as pd
import numpy as np
import time

path = '/neurospin/imagen/workspace/connectogram_examples/BL/'
path_mask = path +'000000106601/probtrackx2/masks.txt'
file_matrix = '/probtrackx2/fdt_network_matrix'
nb_subjects =  959
nb_sigma = 2

## Extract name pf the regions ###
df = pd.read_csv(path_mask, header=None)
clist = df[0].tolist()
areas = []
for j in range(len(clist)):
    m = re.search('/volatile/imagen/connectogram/new_results/BL/000000106601/.*/(.+?).nii.gz', clist[j])
    if m:
        areas.append(m.group(1))

## Extract connectivity matrix of each subject ##
count = 0
t = time.time()
connectivity_matrix = np.zeros((nb_subjects, len(areas)*len(areas)))
for directory in glob.glob(os.path.join(path,'*')):
    if os.path.isdir(directory) and count >= 0:
        connectivity_matrix[count,:] = np.loadtxt(directory+file_matrix).flatten()
        count += 1
elapsed = time.time() - t
print "Elapsed time to create connectivity matrix " + str(elapsed)


names = ['Left-Accumbens-area','Left-Pallidum']
names = ['right-medialorbitofrontal','Right-Accumbens-area']
names =['right-precentral','right-postcentral']
names = ['Right-Amygdala', 'Right-Hippocampus']
index = [areas.index(names[0]),areas.index(names[1])]


import matplotlib.pyplot as plt
import matplotlib as mpl
import math

label_size = 22
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size 
text_size = 26

number_bins = 100


n, bins, patches = plt.hist(connectivity_matrix[:,index[0]*86+index[1]], number_bins, facecolor='blue', normed=True, label='men')
