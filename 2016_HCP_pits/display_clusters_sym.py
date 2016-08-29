"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import json, os
import numpy as np
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

temp_file_s_ids='/home/yl247234/s_ids_lists/s_ids.json'
with open(temp_file_s_ids, 'r') as f:
    data = json.load(f)
s_ids  = list(json.loads(data))

# Note in Fourth also need to modify inputs for thresholds
SYMMETRIC = True
sides = ['R', 'L']
database = "hcp"
database_parcel  = "hcp"
path0 = '/media/yl247234/SAMSUNG/'+database+'/Freesurfer_mesh_database/'
#path0 = '/media/yl247234/SAMSUNG/'+database+'/databaseBV/'
if "Freesurfer" in path0:
    if SYMMETRIC:
        OUTPUT = '/home/yl247234/Images/final_snap_sym/group_'+database_parcel+'_Freesurfer_new/'
    path_parcels = '/media/yl247234/SAMSUNG/'+database_parcel+'/Freesurfer_mesh_database/'
else:
    if SYMMETRIC:
        OUTPUT = '/home/yl247234/Images/final_snap_sym/group_'+database_parcel+'_BV/'
    path_parcels = '/media/yl247234/SAMSUNG/'+database_parcel+'/databaseBV/'
if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

if "Freesurfer" in path0:
    if SYMMETRIC:
        INPUT  = '/neurospin/brainomics/2016_HCP/new_distrib/distribution_sym_DPF_'+database_parcel+'_Freesurfer_new/'
else:
    if SYMMETRIC:
        INPUT  = '/neurospin/brainomics/2016_HCP/new_distrib/distribution_sym_DPF_'+database_parcel+'_BV/'

meshes = {}
tex_mesh = {}
windows = {} 
parcels = {}
parcels_tex = {}
tex_pits = {}
tex_mesh_pits = {}

# First we create snapshots for the group cluster
for side in sides:
    if SYMMETRIC:
        sid = 'L'
    else:
        sid = side
    if "Freesurfer" in path0:
        template_mesh  = '/neurospin/brainomics/folder_gii/'+sid.lower()+'h.inflated.white.gii'
    else:
        template_mesh  = '/neurospin/imagen/workspace/cati/templates/average_'+sid+'mesh_BL.gii'
    meshes[side] = ana.loadObject(template_mesh)
    windows[side] = ana.createWindow('3D', geometry=[0, 0, 584, 584])
    if SYMMETRIC:
        file_parcels_on_atlas = path_parcels +'pits_density_update/clusters_total_average_pits_smoothed0.7_60_sym_dist15.0_area100.0_ridge2.0.gii'
    else:
        file_parcels_on_atlas = path_parcels +'pits_density_update/clusters_'+side+'_average_pits_smoothed0.7_60_dist15.0_area100.0_ridge2.0.gii'
    parcels[side] = ana.loadObject(file_parcels_on_atlas)
    parcels[side].setPalette('pastel-256')
    parcels_tex[side] = ana.fusionObjects([meshes[side], parcels[side]], method='FusionTexSurfMethod')
    if "Freesurfer" in path0:
        ref = ana.createReferential()
        tr = ana.createTransformation([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1] ,ref, ana.centralReferential())
        parcels_tex[side].assignReferential(ref)
        parcels_tex[side].setMaterial(front_face='counterclockwise')
    updateWindow(windows[side], parcels_tex[side])
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
"""
        ana.execute('WindowConfig', windows=[windows[side]], snapshot=OUTPUT+'snapshot_clusters_'+side+'_'+sd+'.jpg')   

# Second we create snapshots for the group density
for side in sides:
    if SYMMETRIC:
        sid = 'L'
    else:
        sid = side
    if "Freesurfer" in path0:
        template_mesh  = '/neurospin/brainomics/folder_gii/'+sid.lower()+'h.inflated.white.gii'
    else:
        template_mesh  = '/neurospin/imagen/workspace/cati/templates/average_'+sid+'mesh_BL.gii'
    meshes[side] = ana.loadObject(template_mesh)
    windows[side] = ana.createWindow('3D', geometry=[0, 0, 584, 584])

    file_parcels_on_atlas = path_parcels +'pits_density_update/total_average_pits_smoothed0.7_60_sym.gii'
    parcels[side] = ana.loadObject(file_parcels_on_atlas)
    parcels[side].setPalette('Purple-Red + Stripes', minVal=0, maxVal=0.15, absoluteMode=True)
    parcels_tex[side] = ana.fusionObjects([meshes[side], parcels[side]], method='FusionTexSurfMethod')
    if "Freesurfer" in path0:
        ref = ana.createReferential()
        tr = ana.createTransformation([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1] ,ref, ana.centralReferential())
        parcels_tex[side].assignReferential(ref)
        parcels_tex[side].setMaterial(front_face='counterclockwise')
    updateWindow(windows[side], parcels_tex[side])
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
        ana.execute('WindowConfig', windows=[windows[side]], snapshot=OUTPUT+'snapshot_density_smooth_'+side+'_'+sd+'.jpg')


# Third we create snapshots with all the single pits projected onto the atlas

for side in sides:
    if SYMMETRIC:
        sid = 'L'
    else:
        sid = side
    pits_data = np.array([])
    for s_id in s_ids:
        if SYMMETRIC:
            file_pits_on_atlas = os.path.join(path0, "hcp", s_id, "t1mri", "BL",
                                              "default_analysis", "segmentation", "mesh",
                                              "surface_analysis_sym", s_id+"_"+side+"white_pits_on_atlas.gii")
        else:
            file_pits_on_atlas = os.path.join(path0, "hcp", s_id, "t1mri", "BL",
                                              "default_analysis", "segmentation", "mesh",
                                              "surface_analysis", s_id+"_"+side+"white_pits_on_atlas.gii")
        if os.path.isfile(file_pits_on_atlas):
            array_pits = gio.read(file_pits_on_atlas).darrays[0].data
            if pits_data.size == 0:
                pits_data = array_pits.astype(int)
            else:
                pits_data += array_pits.astype(int)
        else:
            print "WARNING " +s_id+" for side "+side+" doesn't exist !"
    if "Freesurfer" in path0:
        template_mesh  = '/neurospin/brainomics/folder_gii/'+sid.lower()+'h.inflated.white.gii'
    else:
        template_mesh  = '/neurospin/imagen/workspace/cati/templates/average_'+sid+'mesh_BL.gii'
    meshes[side] = ana.loadObject(template_mesh)
    windows[side] = ana.createWindow('3D', geometry=[0, 0, 584, 584])
    aims_pits = aims.TimeTexture(dtype='FLOAT')
    aims_pits[0].assign(pits_data)
    tex_pits[side] = ana.toAObject(aims_pits)
    tex_mesh_pits[side] = ana.fusionObjects([meshes[side], tex_pits[side]], method='FusionTexSurfMethod')
    tex_pits[side].setPalette('actif_ret')
    if "Freesurfer" in path0:
        ref = ana.createReferential()
        tr = ana.createTransformation([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1] ,ref, ana.centralReferential())
        tex_mesh_pits[side].assignReferential(ref)
        tex_mesh_pits[side].setMaterial(front_face='counterclockwise')
    updateWindow(windows[side], tex_mesh_pits[side])
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
        ana.execute('WindowConfig', windows=[windows[side]], snapshot=OUTPUT+'snapshot_density_'+side+'_'+sd+'.jpg')


# Fourth we show the effect of the auto-thresholding
for side in sides:
    if SYMMETRIC:
        sid = 'L'
    else:
        sid = side
    if SYMMETRIC:
        file_parcels_on_atlas = path_parcels +'pits_density_update/clusters_total_average_pits_smoothed0.7_60_sym_dist15.0_area100.0_ridge2.0.gii'
    else:
        file_parcels_on_atlas = path_parcels +'pits_density_update/clusters_'+side+'_average_pits_smoothed0.7_60_dist15.0_area100.0_ridge2.0.gii'
    array_parcels = gio.read(file_parcels_on_atlas).darrays[0].data
    parcels_org0 =  np.unique(array_parcels)
    parcels_org = parcels_org0[1:]
    pits_data = np.array([])
    INPUT2 = INPUT+side+'/'
    thresholds = np.loadtxt(INPUT2+'thresholds'+side+'.txt')
    for s_id in s_ids:
        if SYMMETRIC:
            file_pits_on_atlas = os.path.join(path0, "hcp", s_id, "t1mri", "BL",
                                              "default_analysis", "segmentation", "mesh",
                                              "surface_analysis_sym", s_id+"_"+side+"white_pits_on_atlas.gii")
            file_DPF_on_atlas = os.path.join(path0, "hcp", s_id, "t1mri", "BL",
                                             "default_analysis", "segmentation",
                                             #s_id+"_"+side+"white_depth_on_atlas.gii")
                                             "mesh", "surface_analysis_sym", ""+s_id+"_"+side+"white_DPF_on_atlas.gii")
        else:
            file_pits_on_atlas = os.path.join(path0, "hcp", s_id, "t1mri", "BL",
                                              "default_analysis", "segmentation", "mesh",
                                              "surface_analysis", s_id+"_"+side+"white_pits_on_atlas.gii")
            file_DPF_on_atlas = os.path.join(path0, "hcp", s_id, "t1mri", "BL",
                                             "default_analysis", "segmentation",
                                             #s_id+"_"+side+"white_depth_on_atlas.gii")
                                             "mesh", "surface_analysis", ""+s_id+"_"+side+"white_DPF_on_atlas.gii")
        
        if os.path.isfile(file_pits_on_atlas) and os.path.isfile(file_DPF_on_atlas):
            array_pits = gio.read(file_pits_on_atlas).darrays[0].data
            array_DPF = gio.read(file_DPF_on_atlas).darrays[0].data
            ind = np.where(parcels_org[0] == array_parcels)
            array_pits[ind] =  array_pits[ind]*0
            for k,parcel in enumerate(parcels_org):
                ind = np.where(parcel == array_parcels)
                array_pits[ind] =  array_pits[ind]*(array_DPF[ind]>thresholds[k])                
            if pits_data.size == 0:
                pits_data = array_pits.astype(int)
            else:
                pits_data += array_pits.astype(int)
        else:
            print "WARNING " +s_id+" for side "+side+" doesn't exist !"
    if "Freesurfer" in path0:
        template_mesh  = '/neurospin/brainomics/folder_gii/'+sid.lower()+'h.inflated.white.gii'
    else:
        template_mesh  = '/neurospin/imagen/workspace/cati/templates/average_'+sid+'mesh_BL.gii'
    meshes[side] = ana.loadObject(template_mesh)
    windows[side] = ana.createWindow('3D', geometry=[0, 0, 584, 584])
    aims_pits = aims.TimeTexture(dtype='FLOAT')
    aims_pits[0].assign(pits_data)
    tex_pits[side] = ana.toAObject(aims_pits)
    tex_mesh_pits[side] = ana.fusionObjects([meshes[side], tex_pits[side]], method='FusionTexSurfMethod')
    tex_pits[side].setPalette('actif_ret')
    if "Freesurfer" in path0:
        ref = ana.createReferential()
        tr = ana.createTransformation([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1] ,ref, ana.centralReferential())
        tex_mesh_pits[side].assignReferential(ref)
        tex_mesh_pits[side].setMaterial(front_face='counterclockwise')
    updateWindow(windows[side], tex_mesh_pits[side])
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
        ana.execute('WindowConfig', windows=[windows[side]], snapshot=OUTPUT+'snapshot_density_thresholded_'+side+'_'+sd+'.jpg')

"""
