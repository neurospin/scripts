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
window = ana.createWindow('3D')

def updateWindow(obj):
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
                         "surface_analysis", ""+ s_id+ "_"+side+"white_DPF.gii")

white_mesh = ana.loadObject(file_white_mesh)

tex = ana.loadObject(file_pits)
tex_mesh = ana.fusionObjects([white_mesh, tex], method='FusionTexSurfMethod')
tex.setPalette('actif_ret')
updateWindow(tex_mesh)


# show/hide palette
#paletteViewer.ShowHidePaletteCallback().doit(anatomist.api.cpp.set_AObjectPtr([sulci.getInternalRep()]))

ana.execute('WindowConfig', windows=[window], cursor_visibility=0)

A= tex.toAimsObject()
U = np.array(A[0])
index_pits2 = []
for j in range(len(U)):
    if U[j] >0:
        index_pits2.append(j)
