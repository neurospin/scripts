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


verbose = False
## INPUTS ###
path0 = '/media/yl247234/SAMSUNG/HCP_from_cluster/databaseBV/'
path = '/media/yl247234/SAMSUNG/HCP_from_cluster/databaseBV/hcp/'
temp_file_s_ids='/home/yl247234/s_ids_lists/s_ids.json'
sides = ['R', 'L']
windows = {}
white_meshes = {}
tex_pits = {}
tex_mesh_pits = {}
## CONSTANTS ##
FILTERING2 = True
FILTERING = False
THRESHOLD_DPF = 1.25

with open(temp_file_s_ids, 'r') as f:
    data = json.load(f)
s_ids  = list(json.loads(data))

## HERE WE VISUALIZE THE PITS ON THE TEMPLATE ##
count = 0
for side in sides:
    file_parcels_on_atlas = path0 +'pits_density/clusters_'+side+'_average_pits_smoothed0.7_60_dist15.0_area100.0_ridge2.0.gii'
    array_parcels = gio.read(file_parcels_on_atlas).darrays[0].data
    parcels_org =  np.unique(array_parcels)
    pits_data = np.array([])
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
            if FILTERING:
                array_pits = array_pits*array_DPF>THRESHOLD_DPF
            elif FILTERING2:
                index_pits = np.nonzero(array_pits)[0]
                parcels = array_parcels[index_pits]
                for k,parcel in enumerate(parcels_org):
                    ind = np.where(parcel == parcels)
                    if verbose:
                        print "LIST OF IND FOR PARCEL: "+str(parcel)
                        print ind
                    # If the subject has pit in this parcel we consider the deepest
                    if ind[0].size:
                        temp = np.zeros(ind[0].size)
                        array_pits[index_pits[ind[0]]] =temp
                        deepest_pit_val = max(array_DPF[index_pits[ind[0]]])
                        if verbose:
                            print "DEEPEST PIT VAL FOUND"
                            print deepest_pit_val
                        # Check if this pit corresponds to the deepest DPF value in the parcel #ie hopefully distinguishing superficial pits
                        ind2 = np.where(parcel == array_parcels)
                        if verbose:
                            print "LIST OF IND2 FOR PARCEL: "+str(parcel)
                            #print ind2[0]
                            print "MAX DPF IN THE PARCEL"
                            print max(array_DPF[ind2[0]])
                        if deepest_pit_val == max(array_DPF[ind2[0]]):
                            count +=1
                            index = np.argmax(array_DPF[index_pits[ind[0]]])
                            if verbose:
                                print "INDEX IN PITS " +str(index)
                                print "INDEX IN ORIGINAL ARRAY"
                                print index_pits[ind[0]][index]
                            array_pits[index_pits[ind[0]][index]] = 1
                            
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
    
    #path_parcels = '/neurospin/imagen/workspace/cati/BVdatabase/'
    #file_parcels_marsAtlas = path_parcels+'cluster_parameters_10.5_150_2_'+side+'_non_sym_smoothed60.gii'
    

    #parcels_data=  gio.read(file_parcels_marsAtlas).darrays[0].data
    
