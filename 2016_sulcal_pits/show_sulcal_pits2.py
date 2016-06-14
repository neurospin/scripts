"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re
import nibabel.gifti.giftiio as gio

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

## Processing to show pits we keep ###
## INPUTS ###
#right 'R' or left 'L'
side = 'R'
PIT_INDEX = 98
INPUT = "/neurospin/brainomics/2016_sulcal_pits/extracting_pits_V0/"
sides = {'R': 'Right',
         'L': 'Left'}

parcels_init = {}
keys_parcels = np.loadtxt('/neurospin/brainomics/2016_sulcal_pits/extracting_pits_V0/'+sides[side]+'_parcels_kept.txt')
s_ids_init = np.loadtxt('/neurospin/brainomics/2016_sulcal_pits/extracting_pits_V0/'+sides[side]+'_s_ids_init.txt')
s_ids_init  = ['%012d' % int(i) for i in s_ids_init]
init_pits = np.loadtxt('/neurospin/brainomics/2016_sulcal_pits/extracting_pits_V0/'+sides[side]+'_init_pits.txt')
j = 0
while sum(init_pits[:j]) < PIT_INDEX:
    j+=1

s_id = s_ids_init[j-1]
parcel = keys_parcels[j-1]
pits = np.loadtxt(INPUT+sides[side]+'/kept/pits/'+side+s_id+'_pits_kept.txt')
parcels = np.loadtxt(INPUT+sides[side]+'/kept/parcels/'+side+s_id+'_parcels_numbers_kept.txt')
index_pit = pits[np.where(parcels == parcel)[0][0]]

# All the subjects with their respective centers should be in
#/neurospin/imagen/src/scripts/psc_tools/psc2_centre.csv
# association centre-number can be found in /neurospin/imagen/RAW/PSC1/, open each centre and read its associated number
centres_number = {'4': 'Berlin','8': 'Dresden','3': 'Dublin','5': 'Hamburg','1': 'London','6': 'Mannheim','2': 'Nottingham','7': 'Paris'}
psc2_centre = np.loadtxt('/neurospin/imagen/src/scripts/psc_tools/psc2_centre.csv', delimiter=',')
all_centres_subjects = {}
for j in range(len(psc2_centre)):
    label = str(int(psc2_centre[j][0]))
    for i in range(12-len(label)):
        label = '0'+label
    all_centres_subjects[label] = centres_number[str(int(psc2_centre[j][1]))]
 
city = all_centres_subjects[s_id]
subject_folder = "/neurospin/imagen/workspace/cati/BVdatabase/"+city+ "/"+ s_id+ "/"

file_pits = os.path.join(subject_folder, "t1mri", "BL",
                         "default_analysis", "segmentation", "mesh",
                         "surface_analysis", ""+ s_id+ "_"+side+"white_pits.gii")

pits_data = gio.read(file_pits).darrays[0].data
pits_data = np.zeros(len(pits_data))
pits_data[index_pit] = 1

file_white_mesh = os.path.join(subject_folder, "t1mri", "BL",
                               "default_analysis", "segmentation", "mesh",
                               ""+ s_id+ "_"+side+"white.gii")
white_mesh = ana.loadObject(file_white_mesh)


### Convert to readable format for
from soma import aims
tex = aims.TimeTexture(dtype='FLOAT')
#tex[0].assign(lh_pval2)
tex[0].assign(pits_data)
tex_pit = ana.toAObject(tex)
tex_mesh_pit = ana.fusionObjects([white_mesh, tex_pit], method='FusionTexSurfMethod')
tex_pit.setPalette('actif_ret')
window = ana.createWindow('3D')
updateWindow(window, tex_mesh_pit)

# show/hide palette
#paletteViewer.ShowHidePaletteCallback().doit(anatomist.api.cpp.set_AObjectPtr([sulci.getInternalRep()]))

ana.execute('WindowConfig', windows=[window], cursor_visibility=0)

"""A= tex_pits.toAimsObject()
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
        index_pits2.append(j)"""
