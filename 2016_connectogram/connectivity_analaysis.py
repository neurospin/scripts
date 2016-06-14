
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
connectivity_matrix = np.zeros((nb_subjects, len(areas),len(areas)))
for directory in glob.glob(os.path.join(path,'*')):
    if os.path.isdir(directory) and count >= 0:
        connectivity_matrix[count,:,:] = np.log(np.loadtxt(directory+file_matrix)+1)
        count += 1
elapsed = time.time() - t
print "Elapsed time " + str(elapsed)

## Statistical analysis ##
means = np.mean(connectivity_matrix, axis=0)
stds = np.std(connectivity_matrix, axis=0)
up = np.asarray(connectivity_matrix) > (means + nb_sigma * stds).reshape(-1,len(areas), len(areas))
down = np.asarray(connectivity_matrix) < (means - nb_sigma * stds).reshape(-1,len(areas), len(areas))
matrix_up =  np.sum(up, axis=0)
matrix_down =  np.sum(down, axis=0)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

text_size = 16
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size 

gs = gridspec.GridSpec(1,2, width_ratios=[8,8], height_ratios=[8])
fig = plt.figure()
ax = []
ax.append(fig.add_subplot(gs[0]))
ax.append(fig.add_subplot(gs[1]))

im = ax[0].imshow(matrix_up/959.0, cmap=cm.gist_rainbow, interpolation="none")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="2%", pad=0.02)
cbar =plt.colorbar(im, cax=cax)
ax[0].set_title('Number of subjects above '+r'$\mu$'+' + '+str(nb_sigma)+r'$\sigma$'+' ',fontsize = text_size+2, fontweight = 'bold')
cbar.set_label('Number of subjects', rotation=270, fontsize = text_size, fontweight = 'bold',labelpad=label_size+2, x=1.01)

im = ax[1].imshow(matrix_down/959.0, cmap=cm.gist_rainbow, interpolation="none")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="2%", pad=0.02)
cbar =plt.colorbar(im, cax=cax)
ax[1].set_title('Number of subjects below of '+r'$\mu$'+' - '+str(nb_sigma)+r'$\sigma$'+' ',fontsize = text_size+2, fontweight = 'bold')
cbar.set_label('Number of subjects', rotation=270, fontsize = text_size, fontweight = 'bold',labelpad=label_size+2, x=1.01)
plt.show()
