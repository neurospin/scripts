"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re

### INPUTS ###
side = 'r' # side of the maillage
sideCAP = 'R' # side of the cluster
# path to the .mat file containing all the results of the MEGHA surf simulation
#mat_results = '/neurospin/brainomics/2016_sulcal_depth/megha/OHBM/more_subjects/1763subjects_covar_GenCit5PCA_ICV_MEGHAcovar_GenCit5PCA_ICV_MEGHA.mat'
mat_results = '/neurospin/brainomics/2016_sulcal_depth/megha/smooth_sulc_same_subjects/covar_GenCitHan5PCA_ICV_MEGHA.mat'
THRESHOLD_PVAL =  4e-1

# To import the environment variable for anatomist
# which -a anatomist
# . ../bv_env.sh
import anatomist.api 
ana = anatomist.api.Anatomist()
# this module path is added only after Anatomist is initialized
import paletteViewer
window = ana.createWindow('3D')

def updateWindow(obj):
    obj.setChanged()
    obj.notifyObservers()


#### DEFINING THE PATH TO GET THE REPRESENTATION OF THE MAILLAGE #####
faverage_pial = '/neurospin/brainomics/folder_gii/'+side+'h.pial.gii'
faverage_Lwhite_inflated = '/neurospin/brainomics/folder_gii/'+side+'h_inflated.white.gii'
faverage_inflated = '/neurospin/brainomics/folder_gii/'+side+'h.inflated.gii'

import scipy.io
mat = scipy.io.loadmat(mat_results)
clus = mat['Clusid'+sideCAP+'h'][0]
Logpval = -np.log10(mat['ClusP'+sideCAP+'h'][0])
thresholded_pval = Logpval > -np.log10(THRESHOLD_PVAL)
for j in range(len(thresholded_pval)):
    Logpval[j] = Logpval[j]*thresholded_pval[j]

Logpval_clus = [Logpval[p-1] if p!=0 else p for p in clus]


### Convert to readable format for
from soma import aims
tex = aims.TimeTexture(dtype='FLOAT')
#tex[0].assign(lh_pval2)
tex[0].assign(Logpval_clus)

atex = ana.toAObject(tex)
lh_white = ana.loadObject(faverage_pial)
inflated = ana.loadObject(faverage_inflated)
white_inflated = ana.loadObject(faverage_Lwhite_inflated)



tex_mesh = ana.fusionObjects([inflated, atex], method='FusionTexSurfMethod')
atex.setPalette('actif_ret', minVal=0, maxVal=2.0, absoluteMode=True)

ref = ana.createReferential()
ref.header()['direct_referential'] = 0 # on n'est meme pas oblige de faire ca
tr = ana.createTransformation([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1] ,ref, ana.centralReferential())
tex_mesh.assignReferential(ref)
tex_mesh.setMaterial(front_face='counterclockwise')
window.addObjects(tex_mesh)
updateWindow(tex_mesh)

# update view so that its state is OK
window.refreshNow()
#updateWindow(tex_mesh)
# show/hide palette
paletteViewer.ShowHidePaletteCallback().doit(anatomist.api.cpp.set_AObjectPtr([atex.getInternalRep()]))

ana.execute('WindowConfig', windows=[window], cursor_visibility=0)
ana.execute('WindowConfig', windows=[window], snapshot='/tmp/snapshot.jpg')

# save palette figure
#gw = window.parent().findChild(paletteViewer.GroupPaletteWidget)
#fig = gw.get(paletteViewer.getObjectId(atex)).findChild(paletteViewer.PaletteWidget).figure
#fig.savefig('/tmp/palette.png')

print "Length average mesh"
lh_aims_white = ana.toAimsObject(white_inflated)
aims_inflated = ana.toAimsObject(inflated)
print len(lh_aims_white.vertex())
print len(aims_inflated.vertex())
print "length Pval"
print len(Logpval_clus)


### USING FREESURFER ###
#freeview -f /neurospin/imagen/BL/processed/freesurfer/fsaverage/surf/rh.inflated:overlay=/volatile/yann/megha/cluster_1000perm/all_covar_cluster_statsLogPvalRh.mgh
#freeview -f /neurospin/imagen/BL/processed/freesurfer/fsaverage/surf/rh.inflated:overlay=/volatile/yann/megha/cluster_1000perm/all_covar_cluster_statsh2Rh.mgh
#freeview -f /neurospin/imagen/BL/processed/freesurfer/fsaverage/surf/lh.inflated:overlay=/volatile/yann/megha/cluster_1000perm/all_covar_cluster_statsLogPvalLh.mgh
#freeview -f /neurospin/imagen/BL/processed/freesurfer/fsaverage/surf/lh.inflated:overlay=/volatile/yann/megha/cluster_1000perm/all_covar_cluster_statsh2Lh.mgh
#freeview -f /neurospin/imagen/BL/processed/freesurfer/fsaverage/surf/lh.inflated:overlay=/volatile/yann/megha/cluster_1000perm_covar_done/covar_GenCit5PCA_ICV_MEGHALogPvalLh.mgh
#freeview -f /neurospin/imagen/BL/processed/freesurfer/fsaverage/surf/lh.inflated:overlay=UPDATEcovar_GenCit5PCA_ICV_MEGHALogPvalLh.mgh
