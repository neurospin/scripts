
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

## Statistical analysis ##
means = np.mean(connectivity_matrix, axis=0)
stds = np.std(connectivity_matrix, axis=0)
up = np.asarray(connectivity_matrix) > (means + nb_sigma * stds).reshape(-1,len(areas)*len(areas))
down = np.asarray(connectivity_matrix) < (means - nb_sigma * stds).reshape(-1,len(areas)*len(areas))
matrix_up =  np.sum(up, axis=0)
matrix_down =  np.sum(down, axis=0)

t = time.time()
scatter_matrix = np.zeros((len(areas)*len(areas), len(areas)*len(areas)))
"""number_cores = 32
from multiprocessing import Process, Queue
def doWork(i,q):
    result = (connectivity_matrix[i,:].reshape(len(areas)*len(areas),1) - means).dot((connectivity_matrix[i,:].reshape(len(areas)*len(areas),1) - means).T)   #put the result in the Queue to return the the calling process
    q.put(result)

#create a Queue to share results
q = Queue()


for j in range(nb_subjects/number_cores):
    p = []
    for i in range(number_cores):
        p.append(Process(target=doWork, args= (i+j*number_cores,q)))
        p[i].start()

    for i in range(number_cores):
        scatter_matrix += q.get(True)
        p[i].join()
        print "Subject number "+ str(i)+ " done"

p = []
for i in range(nb_subjects/number_cores*number_cores,nb_subjects):
    p.append(Process(target=doWork, args= (i+j*nb_subjects/number_cores,q)))
    p[i].start()

for i in range(nb_subjects/number_cores*number_cores,nb_subjects):
    scatter_matrix += q.get(True)
    p[i].join()
    print "Subject number "+ str(i)+ " done"
"""
# Previous code for which Elapsed time to create scatter matrix 60244.3134298
"""for j in range(nb_subjects):
    scatter_matrix += (connectivity_matrix[i,:].reshape(len(areas)*len(areas),1) - means).dot((connectivity_matrix[i,:].reshape(len(areas)*len(areas),1) - means).T)
"""
# Now just load the result
scatter_matrix = np.loadtxt('/neurospin/brainomics/2016_connectogram/scatter_matrix3.txt')

elapsed = time.time() - t
print "Elapsed time to create scatter matrix " + str(elapsed)   

t = time.time()
# eigenvectors and eigenvalues for the from the scatter matrix
# Previous code Elapsed time to compute eigenvectors 684.365755081
#eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix) 
eig_val_sc = np.loadtxt('/neurospin/brainomics/2016_connectogram/eig_val_sc3.txt').view(complex)
eig_vec_sc = np.loadtxt('/neurospin/brainomics/2016_connectogram/eig_vec_sc3.txt').view(complex)
elapsed = time.time() - t
print "Elapsed time to compute eigenvectors " + str(elapsed)  
                
# Make a list of (eigenvalue, eigenvector) tuples
"""eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i])
             for i in range(len(eig_val_sc))]"""

eig_pairs_bis = [(np.abs(eig_val_sc[i]), i) for i in range(len(eig_val_sc))]
# Sort the (eigenvalue, eigenvector) tuples from high to low

eig_pairs_bis.sort()
eig_pairs_bis.reverse()

matrix_w = np.hstack((eig_vec_sc[eig_pairs_bis[0][1]].reshape(len(areas)*len(areas),1),eig_vec_sc[eig_pairs_bis[1][1]].reshape(len(areas)*len(areas),1)))

transformed = matrix_w.T.dot(connectivity_matrix.T)



from matplotlib import pyplot as plt
plt.plot(transformed[0,:], transformed[1,:],
         'o', markersize=7, color='blue', alpha=0.5, label='Subjects')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed space - subject positions')
#plt.show()


pca_component1 = np.zeros((len(areas), len(areas)))
pca_component2 = np.zeros((len(areas), len(areas)))
for j in range(len(areas)):
    for i in range(len(areas)):
        pca_component1[j,i] = np.abs(eig_vec_sc[eig_pairs_bis[0][1]][j*len(areas)+i])
        pca_component2[j,i] = np.abs(eig_vec_sc[eig_pairs_bis[1][1]][j*len(areas)+i])

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

im = ax[0].imshow(pca_component1, cmap=cm.gist_rainbow, interpolation="none")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="2%", pad=0.02)
cbar =plt.colorbar(im, cax=cax)
ax[0].set_title('Principal component 1',fontsize = text_size+2, fontweight = 'bold')
cbar.set_label('% of the direction', rotation=270, fontsize = text_size, fontweight = 'bold',labelpad=label_size+2, x=1.01)

im = ax[1].imshow(pca_component2, cmap=cm.gist_rainbow, interpolation="none")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="2%", pad=0.02)
cbar =plt.colorbar(im, cax=cax)
ax[1].set_title('Principal component 2',fontsize = text_size+2, fontweight = 'bold')
cbar.set_label('% of the direction', rotation=270, fontsize = text_size, fontweight = 'bold',labelpad=label_size+2, x=1.01)
plt.show()


t = time.time()
# Previously Elapsed time to create distance matrix 3755.19680905
"""distance_matrix = np.zeros((len(areas)*len(areas), len(areas)*len(areas)))
for j in range(len(areas)*len(areas)):
    for i in range(len(areas)*len(areas)):
        distance_matrix[j,i] = np.sqrt(np.sum(np.square(connectivity_matrix[:,i]-connectivity_matrix[:,j])))
    # Only print by row else too many prints
    print "Link position: ("+ str(j)+","+str(i)+")"""
distance_matrix = np.loadtxt('/neurospin/brainomics/2016_connectogram/distance_matrix.txt')
elapsed = time.time() - t
print "Elapsed time to create distance matrix " + str(elapsed)   

