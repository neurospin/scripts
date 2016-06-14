"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re

## INPUTS ###
#right 'R' or left 'L'
side = 'L'

# To import the environment variable for anatomist
# which -a anatomist
# . ../bv_env.sh
import anatomist.api 
ana = anatomist.api.Anatomist()
# this module path is added only after Anatomist is initialized
import paletteViewer

def updateWindow(window, obj):
    window.addObjects(obj)
    obj.setChanged()
    obj.notifyObservers()

sides = {'R': 'right',
         'L': 'left'}
# Load the right hemisphere folds mesh, associated graph and cortical surface 
s_id_y = '000000106871'
s_id_antoine = '000001311901'
city = 'Dresden/'
#city = 'Mannheim/'#
s_id = s_id_y
feature = '_depthMax'

#### DEFINING THE PATH TO GET THE REPRESENTATION OF THE MAILLAGE #####

subject_folder = "/neurospin/imagen/workspace/cati/BVdatabase/"+city+ s_id+ "/"

file_white_mesh = os.path.join(subject_folder, "t1mri", "BL",
                               "default_analysis", "segmentation", "mesh",
                               ""+ s_id+ "_"+side+"white.gii")

file_pits = os.path.join(subject_folder, "t1mri", "BL",
                         "default_analysis", "segmentation", "mesh",
                         "surface_analysis", ""+ s_id+ "_"+side+"white_pits_on_atlas.gii")

file_parcels_marsAtlas = os.path.join(subject_folder, "t1mri", "BL",
                         "default_analysis", "segmentation", "mesh",
                         "surface_analysis", ""+ s_id+ "_"+side+"white_parcels_marsAtlas.gii")
path = '/neurospin/imagen/workspace/cati/BVdatabase/'

file_parcels_marsAtlas = path+side+'_clusters_default_parameters.gii'
file_white_mesh = '/neurospin/imagen/workspace/cati/BVdatabase/average_'+side+'mesh_BL.gii'
white_mesh = ana.loadObject(file_white_mesh)

window = ana.createWindow('3D')
tex_pits = ana.loadObject(file_pits)
tex_mesh_pits = ana.fusionObjects([white_mesh, tex_pits], method='FusionTexSurfMethod')
tex_pits.setPalette('actif_ret')
updateWindow(window, tex_mesh_pits)

window2 = ana.createWindow('3D')
tex_parcels = ana.loadObject(file_parcels_marsAtlas)
tex_mesh_parcels = ana.fusionObjects([white_mesh, tex_parcels], method='FusionTexSurfMethod')
tex_parcels.setPalette('pastel-256')
updateWindow(window2, tex_mesh_parcels)

# show/hide palette
#paletteViewer.ShowHidePaletteCallback().doit(anatomist.api.cpp.set_AObjectPtr([sulci.getInternalRep()]))

ana.execute('WindowConfig', windows=[window], cursor_visibility=0)
ana.execute('WindowConfig', windows=[window2], cursor_visibility=0)

A= tex_pits.toAimsObject()
U = np.array(A[0])
index_pits = []
for j in range(len(U)):
    if U[j] >0:
        index_pits.append(j)

A= tex_parcels.toAimsObject()
U2 = np.array(A[0])
index_pits2 = []
for j in range(len(U2)):
    if U2[j] == 1:
        index_pits2.append(j)

max = 0
count = 0
for j in range(2571):
    nb= len(U2) -np.count_nonzero(U2-j)
    if nb > max:
        max = nb
    if nb > 1:
        count +=1
        print "For parcel "+str(j)+ "count = "+ str(nb)

U3 = [k if k in range(2571) else 0 for k in U2]
from soma import aims
tex = aims.TimeTexture(dtype='FLOAT')
tex[0].assign(U3)
tex_parcels2 = ana.toAObject(tex)
window3 = ana.createWindow('3D')
tex_mesh_parcels3 = ana.fusionObjects([white_mesh, tex_parcels2], method='FusionTexSurfMethod')
tex_parcels2.setPalette('pastel-256')
updateWindow(window3, tex_mesh_parcels3)
