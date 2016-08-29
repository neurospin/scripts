"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re, json, argparse
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
INTERVAL2 = [-np.log10(1e-1), 8]
colorbar = 'Yellow-red-fusion'
colorbar2 = 'rainbow2-fusion'
#colorbar = 'actif_ret'
CC_ANALYSIS = False # Deprecated soon

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('-f', '--feature', type=str,
                        help="Features to measure")
    parser.add_argument('-s', '--symmetric', type=int,
                        help="Boolean, need to give 0 for False and another int for True " 
                        "Specify if symmetric template is used or not")
    parser.add_argument('-d', '--database', type=str,
                        help='Data base from which we take the cluster')
    parser.add_argument('-t', '--feature_threshold', type=str,
                        help='Specify the feature used for thresholding either sulc or DPF')
    options = parser.parse_args()
    ## INPUTS ###
    feature = options.feature
    SYMMETRIC = bool(int(options.symmetric))
    database_parcel  = options.database

    feature_threshold = options.feature_threshold
    
    """
    feature = 'DPF'
    database_parcel = 'hcp'
    SYMMETRIC = True
    """
    path_parcels = '/media/yl247234/SAMSUNG/'+database_parcel+'/Freesurfer_mesh_database/'

    if "Freesurfer" in path_parcels:
        if SYMMETRIC:
            OUTPUT = '/home/yl247234/Images/final_snap_sym_threshold_'+feature_threshold+'/group_'+database_parcel+'_Freesurfer_new/'
            INPUT = '/neurospin/brainomics/2016_HCP/new_dictionaries_herit_threshold_'+feature_threshold+'/pits_sym_'+feature+'_'+database_parcel+'_Freesurfer_new'
        else:
            OUTPUT = '/home/yl247234/Images/final_snap_threshold_'+feature_threshold+'/group_'+database_parcel+'_Freesurfer_new/'
            INPUT = '/neurospin/brainomics/2016_HCP/new_dictionaries_herit_threshold_'+feature_threshold+'/pits_'+feature+'_'+database_parcel+'_Freesurfer_new'
    else:
        if SYMMETRIC:
            INPUT = '/neurospin/brainomics/2016_HCP/new_dictionaries_herit_threshold_'+feature_threshold+'/pits_sym_'+feature+'_'+database_parcel+'_BV'
            OUTPUT = '/home/yl247234/Images/final_snap_sym_threshold_'+feature_threshold+'/group_'+database_parcel+'_BV/'
        else:
            OUTPUT = '/home/yl247234/Images/final_snap_threshold_'+feature_threshold+'/group_'+database_parcel+'_BV/'
            INPUT = '/neurospin/brainomics/2016_HCP/new_dictionaries_herit_threshold_'+feature_threshold+'/pits_'+feature+'_'+database_parcel+'_BV'
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)


    if CC_ANALYSIS:
        INPUT = INPUT+"CC_"
        pheno0 = 'case_control_'
    else:
        pheno0 = feature+'_'

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
        if "Freesurfer" in path_parcels:
            if "sym" in path_parcels:
                template_mesh  = '/neurospin/brainomics/folder_gii/'+sid.lower()+'h.inflated.white.gii'
            else:
                template_mesh  = '/neurospin/brainomics/folder_gii/'+sid.lower()+'h.inflated.white.gii'
        else:
            template_mesh  = '/neurospin/imagen/workspace/cati/templates/average_'+sid+'mesh_BL.gii'
        meshes[side] = ana.loadObject(template_mesh)
        windows[side] = ana.createWindow('3D', geometry=[0, 0, 584, 584])
        if SYMMETRIC:
            file_parcels_on_atlas = path_parcels +'pits_density_update/clusters_total_average_pits_smoothed0.7_60_sym_dist15.0_area100.0_ridge2.0.gii'
        else:
            file_parcels_on_atlas = path_parcels +'pits_density_update/clusters_'+side+'_average_pits_smoothed0.7_60_dist15.0_area100.0_ridge2.0.gii'
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
        if "Freesurfer" in path_parcels:
            ref = ana.createReferential()
            tr = ana.createTransformation([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1] ,ref, ana.centralReferential())
            tex_mesh[side].assignReferential(ref)
            tex_mesh[side].setMaterial(front_face='counterclockwise')
        pits_tex[side].setPalette(colorbar, minVal=INTERVAL[0], maxVal=INTERVAL[1], absoluteMode=True)
        updateWindow(windows[side], tex_mesh[side])              
        #paletteViewer.ShowHidePaletteCallback().doit(anatomist.api.cpp.set_AObjectPtr([pits_tex[side].getInternalRep()]))
        ana.execute('WindowConfig', windows=[windows[side]], cursor_visibility=0)
        if (side == "R" and "Freesurfer" not in path_parcels and not SYMMETRIC) or (side == "L" and "Freesurfer" in path_parcels and not SYMMETRIC) or (SYMMETRIC and "Freesurfer" in path_parcels):
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


    for side in sides:
        if SYMMETRIC:
            sid = 'L'
        else:
            sid = side 
        pheno = pheno0+sds[side]
        if "Freesurfer" in path_parcels:
            template_mesh  = '/neurospin/brainomics/folder_gii/'+sid.lower()+'h.inflated.white.gii'
        else:
            template_mesh  = '/neurospin/imagen/workspace/cati/templates/average_'+sid+'mesh_BL.gii'
        meshes[side] = ana.loadObject(template_mesh)
        windows[side] = ana.createWindow('3D', geometry=[0, 0, 584, 584])
        if SYMMETRIC:
            file_parcels_on_atlas = path_parcels +'pits_density_update/clusters_total_average_pits_smoothed0.7_60_sym_dist15.0_area100.0_ridge2.0.gii'
        else:
            file_parcels_on_atlas = path_parcels +'pits_density_update/clusters_'+side+'_average_pits_smoothed0.7_60_dist15.0_area100.0_ridge2.0.gii'
        array_parcels = gio.read(file_parcels_on_atlas).darrays[0].data
        array_pval = np.zeros(len(array_parcels))
        parcels_org =  np.unique(array_parcels)
        for parcel in parcels_org:
            if dict_pval[pheno].has_key(str(int(parcel))):
                ind = np.where(array_parcels == parcel)[0]
                #if dict_pval[pheno][str(int(parcel))] < 0.05/1000:
                array_pval[ind] = -np.log10(dict_pval[pheno][str(int(parcel))])
        tex = aims.TimeTexture(dtype='FLOAT')
        tex[0].assign(array_pval)                  
        pits_tex[side] = ana.toAObject(tex)
        tex_mesh[side] = ana.fusionObjects([meshes[side], pits_tex[side]], method='FusionTexSurfMethod')
        if "Freesurfer" in path_parcels:
            ref = ana.createReferential()
            tr = ana.createTransformation([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1] ,ref, ana.centralReferential())
            tex_mesh[side].assignReferential(ref)
            tex_mesh[side].setMaterial(front_face='counterclockwise')
        pits_tex[side].setPalette(colorbar2, minVal=INTERVAL2[0], maxVal=INTERVAL2[1], absoluteMode=True)
        updateWindow(windows[side], tex_mesh[side])              
        #paletteViewer.ShowHidePaletteCallback().doit(anatomist.api.cpp.set_AObjectPtr([pits_tex[side].getInternalRep()]))
        ana.execute('WindowConfig', windows=[windows[side]], cursor_visibility=0)
        if (side == "R" and "Freesurfer" not in path_parcels and not SYMMETRIC) or (side == "L" and "Freesurfer" in path_parcels and not SYMMETRIC) or (SYMMETRIC and "Freesurfer" in path_parcels):
            view_quaternions = {'intern' : [0.5, 0.5, 0.5, 0.5],
                                'extern' : [0.5, -0.5, -0.5, 0.5]}
        else:
            view_quaternions = {'extern' : [0.5, 0.5, 0.5, 0.5],
                                'intern' : [0.5, -0.5, -0.5, 0.5]}
        for sd in view_quaternions.keys():
            q = aims.Quaternion(view_quaternions[sd])
            windows[side].camera(view_quaternion=view_quaternions[sd],
                                 zoom=0.65)

            ana.execute('WindowConfig', windows=[windows[side]], snapshot=OUTPUT+'snapshot_pval_'+pheno+'_'+sd+'.jpg')
        paletteViewer.ShowHidePaletteCallback().doit(anatomist.api.cpp.set_AObjectPtr([pits_tex[side].getInternalRep()]))

