"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re, json
import nibabel.gifti.giftiio as gio
from soma import aims
import anatomist.api 
ana = anatomist.api.Anatomist()
import paletteViewer

def updateWindow(window, obj):
    window.addObjects(obj)
    obj.setChanged()
    obj.notifyObservers()
    ana.execute('WindowConfig', windows=[window], cursor_visibility=0)
INTERVAL = [0,1.0]
colorbar = 'Yellow-red-fusion'
#colorbar = 'actif_ret'
CC_ANALYSIS = False
SYMMETRIC = False
database_parcel = 'all_data'
path_parcels = '/media/yl247234/SAMSUNG/'+database_parcel+'/databaseBV/'

if "Freesurfer" in path_parcels:
    path0 = '/media/yl247234/SAMSUNG/'+database_parcel+'/Freesurfer_mesh_database/'
    if SYMMETRIC:
        OUTPUT = '/home/yl247234/Images/new_snap_depth_sym/group_'+database_parcel+'_Freesurfer/'
        INPUT = '/neurospin/brainomics/2016_HCP/dictionaries_herit/pits_sym_Depth_'+database_parcel+'_Freesurfer'
    else:
        OUTPUT = '/home/yl247234/Images/new_snap_depth/group_'+database_parcel+'_Freesurfer/'
        INPUT = '/neurospin/brainomics/2016_HCP/dictionaries_herit/pits_Depth_'+database_parcel+'_Freesurfer'
else:    
    path0 = '/media/yl247234/SAMSUNG/'+database_parcel+'/databaseBV/'
    if SYMMETRIC:
        INPUT = '/neurospin/brainomics/2016_HCP/dictionaries_herit/pits_sym_Depth_'+database_parcel+'_BV'
        OUTPUT = '/home/yl247234/Images/new_snap_depth_sym/group_'+database_parcel+'_BV/'
    else:
        OUTPUT = '/home/yl247234/Images/new_snap_depth/group_'+database_parcel+'_BV/'
        INPUT = '/neurospin/brainomics/2016_HCP/dictionaries_herit/pits_Depth_'+database_parcel+'_BV'


if CC_ANALYSIS:
    INPUT = INPUT+"CC_"
    pheno0 = 'case_control_'
else:
    pheno0 = 'DPF_'

with open(INPUT+'pval_dict.json', 'r') as f:
    data = json.load(f)
dict_pval = json.loads(data)
with open(INPUT+'h2_dict.json', 'r') as f:
    data = json.load(f)
dict_h2 = json.loads(data)

sides = ['R', 'L']
sds = {'R' : 'Right', 'L': 'Left'}

meshes = {}
tex_mesh = {}
windows = {} 
pits_tex = {}
for side in sides:
    if SYMMETRIC:
        sid = 'L'
    else:
        sid = side 
    pheno = pheno0+sds[side]
    if "Freesurfer" in path0:
        template_mesh  = '/neurospin/brainomics/folder_gii/'+sid.lower()+'h.inflated.white.gii'
    else:
        template_mesh  = '/neurospin/imagen/workspace/cati/templates/average_'+sid+'mesh_BL.gii'
    meshes[side] = ana.loadObject(template_mesh)
    windows[side] = ana.createWindow('3D', geometry=[0, 0, 584, 584])
    if SYMMETRIC:
        file_parcels_on_atlas = path_parcels +'pits_density/clusters_total_average_pits_smoothed0.7_60_sym_dist15.0_area100.0_ridge2.0.gii'
    else:
        file_parcels_on_atlas = path_parcels +'pits_density/clusters_'+side+'_average_pits_smoothed0.7_60_dist15.0_area100.0_ridge2.0.gii'
    array_parcels = gio.read(file_parcels_on_atlas).darrays[0].data
    array_h2 = np.zeros(len(array_parcels))
    parcels_org =  np.unique(array_parcels)
    for parcel in parcels_org:
        if dict_h2[pheno].has_key(str(int(parcel))):
            ind = np.where(array_parcels == parcel)[0]
            array_h2[ind] = dict_h2[pheno][str(int(parcel))]
    tex = aims.TimeTexture(dtype='FLOAT')
    tex[0].assign(array_h2)                  
    pits_tex[side] = ana.toAObject(tex)
    tex_mesh[side] = ana.fusionObjects([meshes[side], pits_tex[side]], method='FusionTexSurfMethod')
    if "Freesurfer" in path0:
        ref = ana.createReferential()
        tr = ana.createTransformation([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1] ,ref, ana.centralReferential())
        tex_mesh[side].assignReferential(ref)
        tex_mesh[side].setMaterial(front_face='counterclockwise')
    pits_tex[side].setPalette(colorbar, minVal=INTERVAL[0], maxVal=INTERVAL[1], absoluteMode=True)
    updateWindow(windows[side], tex_mesh[side])              
    #paletteViewer.ShowHidePaletteCallback().doit(anatomist.api.cpp.set_AObjectPtr([pits_tex[side].getInternalRep()]))
    ana.execute('WindowConfig', windows=[windows[side]], cursor_visibility=0)
    if (side == "R" and "Freesurfer" not in path0 and not SYMMETRIC) or (side == "L" and "Freesurfer" in path0 and not SYMMETRIC) or (SYMMETRIC and "Freesurfer" in path0):
        view_quaternions = {'intern' : [0.5, 0.5, 0.5, 0.5],
                            'extern' : [0.5, -0.5, -0.5, 0.5]}
    else:
        view_quaternions = {'extern' : [0.5, 0.5, 0.5, 0.5],
                            'intern' : [0.5, -0.5, -0.5, 0.5]}
    for sd in view_quaternions.keys():
        q = aims.Quaternion(view_quaternions[sd])
        windows[side].camera(view_quaternion=view_quaternions[sd],
                             zoom=0.65)
        
        ana.execute('WindowConfig', windows=[windows[side]], snapshot=OUTPUT+'snapshot_'+pheno+'_'+sd+'.jpg')
