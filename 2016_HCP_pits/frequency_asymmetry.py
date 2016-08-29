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
INTERVAL = [0.5,1.0]
colorbar = 'Yellow-red-fusion'
colorbar2 = 'rainbow2-fusion'
database_parcel = 'imagen'
path_parcels = '/media/yl247234/SAMSUNG/'+database_parcel+'/databaseBV/'
OUTPUT = '/home/yl247234/Images/new_snap_sym/group_'+database_parcel+'_BV/'

sides = ['R', 'L']
pheno_dir = '/neurospin/brainomics/2016_HCP/pheno_pits_sym_DPF_'+database_parcel+'_BV/'


dict_freq = {}
for side in sides:
    dict_freq[side] = {}
    filename = pheno_dir+"case_control/all_pits_side"+side+".csv"
    df = pd.read_csv(filename)
    nb_subj = df.shape[0]
    for col in df.columns:
        if col != 'IID':
            df[col] = df[col]-1
            nb_pits = np.count_nonzero(df[col])
            freq = float(nb_pits)/nb_subj
            dict_freq[side][col] = freq#*(freq>0.5)
dict_freq["asym"] = {}
for col in df.columns:
    if col != 'IID':
        dict_freq["asym"][col] = 2*(dict_freq["L"][col]-dict_freq["R"][col])/(dict_freq["L"][col]+dict_freq["R"][col])
        dict_freq["L"][col] = dict_freq["L"][col]*(dict_freq["L"][col]>0.5)
        dict_freq["R"][col] = dict_freq["R"][col]*(dict_freq["R"][col]>0.5)

print "NUMBER OF CLUSTER CONSIDERED LEFT " + str(np.count_nonzero(dict_freq["L"].values()))
print "NUMBER OF CLUSTER CONSIDERED RIGHT " + str(np.count_nonzero(dict_freq["R"].values()))

path0 = '/media/yl247234/SAMSUNG/'+database_parcel+'/databaseBV/'

sides = ['R', 'L', "asym"]
meshes = {}
tex_mesh = {}
windows = {} 
pits_tex = {}
for side in sides:
    if "Freesurfer" in path0:
        template_mesh  = '/neurospin/brainomics/folder_gii/lh.inflated.white.gii'
    else:
        template_mesh  = '/neurospin/imagen/workspace/cati/templates/average_Lmesh_BL.gii'
    meshes[side] = ana.loadObject(template_mesh)
    windows[side] = ana.createWindow('3D', geometry=[0, 0, 584, 584])
    file_parcels_on_atlas = path_parcels +'pits_density/clusters_total_average_pits_smoothed0.7_60_sym_dist15.0_area100.0_ridge2.0.gii'
    array_parcels = gio.read(file_parcels_on_atlas).darrays[0].data
    array_freq = np.zeros(len(array_parcels))
    parcels_org =  np.unique(array_parcels)
    parcels_org = parcels_org[1:]
    for parcel in parcels_org:
        ind = np.where(array_parcels == parcel)[0]
        array_freq[ind] = dict_freq[side]["Parcel_"+str(int(parcel))]
    tex = aims.TimeTexture(dtype='FLOAT')
    tex[0].assign(array_freq)                  
    pits_tex[side] = ana.toAObject(tex)
    tex_mesh[side] = ana.fusionObjects([meshes[side], pits_tex[side]], method='FusionTexSurfMethod')
    if "Freesurfer" in path0:
        ref = ana.createReferential()
        tr = ana.createTransformation([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1] ,ref, ana.centralReferential())
        tex_mesh[side].assignReferential(ref)
        tex_mesh[side].setMaterial(front_face='counterclockwise')
    updateWindow(windows[side], tex_mesh[side])              
    if side != "asym":
        pits_tex[side].setPalette(colorbar, minVal=INTERVAL[0], maxVal=INTERVAL[1], absoluteMode=True)
    else:
        pits_tex[side].setPalette(colorbar2, minVal=-0.25, maxVal=0.25, absoluteMode=True)
    ana.execute('WindowConfig', windows=[windows[side]], cursor_visibility=0)
    view_quaternions = {'intern' : [0.5, 0.5, 0.5, 0.5],
                        'extern' : [0.5, -0.5, -0.5, 0.5]}
    for sd in view_quaternions.keys():
        q = aims.Quaternion(view_quaternions[sd])
        windows[side].camera(view_quaternion=view_quaternions[sd],
                             zoom=0.65)
        
        ana.execute('WindowConfig', windows=[windows[side]], snapshot=OUTPUT+'snapshot_frequency_'+side+'_'+sd+'.jpg')
    paletteViewer.ShowHidePaletteCallback().doit(anatomist.api.cpp.set_AObjectPtr([pits_tex[side].getInternalRep()]))
