"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re
from soma import aims
## INPUTS ###
#right 'R' or left 'L'
side = 'R'

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

#### DEFINING THE PATH TO GET THE REPRESENTATION OF THE MAILLAGE #####
file_white_mesh = '/neurospin/imagen/workspace/cati/BVdatabase/average_'+side+'mesh_BL.gii'
INPUT = '/neurospin/imagen/workspace/cati/BVdatabase/'
filename = side+'_average_pits_smoothed.txt'
#file_pits = '/neurospin/imagen/workspace/cati/BVdatabase/'+side+'_average_pits_smoothed.gii'

pits_data = np.loadtxt(INPUT+filename)


white_mesh = ana.loadObject(file_white_mesh)

window = ana.createWindow('3D')
tex = aims.TimeTexture(dtype='FLOAT')
tex[0].assign(pits_data)
tex_pits = ana.toAObject(tex)
#tex_pits = ana.loadObject(file_pits)
tex_mesh_pits = ana.fusionObjects([white_mesh, tex_pits], method='FusionTexSurfMethod')
tex_pits.setPalette('RAINBOW')
"""ref = ana.createReferential()
ref.header()['direct_referential'] = 0 # on n'est meme pas oblige de faire ca
tr = ana.createTransformation([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1] ,ref, ana.centralReferential())
tex_mesh_pits.assignReferential(ref)
tex_mesh_pits.setMaterial(front_face='counterclockwise')"""
updateWindow(window, tex_mesh_pits)
window.refreshNow()

#paletteViewer.ShowHidePaletteCallback().doit(anatomist.api.cpp.set_AObjectPtr([tex_pits.getInternalRep()]))

ana.execute('WindowConfig', windows=[window], cursor_visibility=0)
