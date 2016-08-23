"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re, json
import nibabel.gifti.giftiio as gio
import time

import anatomist.api 
ana = anatomist.api.Anatomist()
import paletteViewer
from soma import aims

def updateWindow(window, obj):
    window.addObjects(obj)
    obj.setChanged()
    obj.notifyObservers()


## INPUTS ###
path0 = '/media/yl247234/SAMSUNG/HCP_from_cluster/databaseBV/'
path = '/media/yl247234/SAMSUNG/HCP_from_cluster/databaseBV/hcp/'
temp_file_s_ids='/home/yl247234/s_ids_lists/s_ids.json'
sides = ['R', 'L']
windows = {}
white_meshes = {}
tex_pits = {}
tex_mesh_pits = {}
with open(temp_file_s_ids, 'r') as f:
    data = json.load(f)
s_ids  = list(json.loads(data))

## HERE WE VISUALIZE THE PITS ON THE TEMPLATE ##
count = 0
for side in sides:
    file_parcels_on_atlas = path0 +'pits_density/clusters_'+side+'_average_pits_smoothed0.7_60_dist15.0_area100.0_ridge2.0.gii'
    array_parcels = gio.read(file_parcels_on_atlas).darrays[0].data
    parcels_org =  np.unique(array_parcels)
    parcels_org = parcels_org[1:]
    pits_data = np.array([])
    INPUT  = '/neurospin/brainomics/2016_HCP/distribution_DPF/'+side+'/'
    thresholds = np.loadtxt(INPUT+'thresholds'+side+'.txt')
    for s_id in s_ids:
        file_pits_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                          "default_analysis", "segmentation", "mesh",
                                          "surface_analysis", ""+s_id+"_"+side+"white_pits_on_atlas.gii")
        file_DPF_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                          "default_analysis", "segmentation", "mesh",
                                          "surface_analysis", ""+s_id+"_"+side+"white_DPF_on_atlas.gii")
        if os.path.isfile(file_pits_on_atlas) and os.path.isfile(file_DPF_on_atlas):
            array_pits = gio.read(file_pits_on_atlas).darrays[0].data
            array_DPF = gio.read(file_DPF_on_atlas).darrays[0].data
            for k,parcel in enumerate(parcels_org):
                ind = np.where(parcel == array_parcels)
                array_pits[ind] =  array_pits[ind]*array_DPF[ind]>thresholds[k]
            if pits_data.size == 0:
                pits_data = array_pits.astype(int)
            else:
                pits_data += array_pits.astype(int)
        else:
            print "WARNING " +s_id+" for side "+side+" doesn't exist !"

    template_mesh  = '/neurospin/imagen/workspace/cati/templates/average_'+side+'mesh_BL.gii'
    white_meshes[side] = ana.loadObject(template_mesh)
    windows[side] = ana.createWindow('3D')
    aims_pits = aims.TimeTexture(dtype='FLOAT')
    aims_pits[0].assign(pits_data)
    tex_pits[side] = ana.toAObject(aims_pits)
    tex_mesh_pits[side] = ana.fusionObjects([white_meshes[side], tex_pits[side]], method='FusionTexSurfMethod')
    tex_pits[side].setPalette('actif_ret')
    updateWindow(windows[side], tex_mesh_pits[side])
    ana.execute('WindowConfig', windows=[windows[side]], cursor_visibility=0)
