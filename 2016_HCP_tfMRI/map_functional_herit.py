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
INTERVAL = [0,0.5]
INTERVAL2 = [-np.log10(1e-1), 10]
colorbar = 'Yellow-red-fusion'
colorbar2 = 'zfun-EosB'
colorbar2 = 'green_yellow_red'
#colorbar2 = 'Green-blue-fusion'
#colorbar = 'actif_ret'

tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'RELATIONAL', 'SOCIAL', 'WM']
COPE_NUMS = [[1,3], [1,3], [1,4], [1,4], [1,3], [1, 22]]
group_path = '/neurospin/brainomics/2016_HCP/functional_analysis/HCP_MMP1.0'

if __name__ == '__main__': 
    filename  ='/media/yl247234/SAMSUNG/HCP_MMP1.0/parcel_names.csv'
    df_labels = pd.read_csv(filename)
    df_labels.index = df_labels['Index']
    parcels_name = [name.replace('\n', '').replace('/', ' ') for name in df_labels['Area Description']]
    df_labels['Area Description'] = parcels_name
   
    sides = ['L', 'R']
  
    for BONF in [180, 360, 1]:
        BONFTHRESHOLD = 5e-2/BONF
        for j, task in enumerate(tasks):
            for i in range(COPE_NUMS[j][0],COPE_NUMS[j][1]+1):
                OUTPUT = '/neurospin/brainomics/2016_HCP/functional_analysis/FIGURES/pheno_mean_value/'+task+'_'+str(i)+'/'
                INPUT = '/neurospin/brainomics/2016_HCP/functional_analysis/herit_dict/pheno_mean_value/'+task+'_'+str(i)

                if not os.path.exists(OUTPUT):
                    os.makedirs(OUTPUT)

                with open(INPUT+'pval_dict.json', 'r') as f:
                    data = json.load(f)
                dict_pval = json.loads(data)
                with open(INPUT+'h2_dict.json', 'r') as f:
                    data = json.load(f)
                dict_h2 = json.loads(data)


                meshes = {}
                tex_mesh = {}
                windows = {} 
                pits_tex = {}
                tex = aims.TimeTexture(dtype='FLOAT')
                
                # Displaying heritability values
                for side in sides:
                    template_mesh  = '/neurospin/brainomics/folder_gii/'+side.lower()+'h.inflated.white.gii'
                    meshes[side] = ana.loadObject(template_mesh)
                    windows[side] = ana.createWindow('3D', geometry=[0, 0, 584, 584])
                    file_parcels_on_atlas = os.path.join(group_path, side+'.fsaverage164k.label.gii')
                    array_parcels = gio.read(file_parcels_on_atlas).darrays[0].data
                    array_h2 = np.zeros(len(array_parcels))
                    parcels_org =  np.unique(array_parcels)
                    parcels_org = parcels_org[1:]
                    for parcel in parcels_org:
                        if dict_h2[side].has_key(str(int(parcel))):
                            if dict_pval[side][str(int(parcel))] < BONFTHRESHOLD:
                                ind = np.where(array_parcels == parcel)[0]
                                array_h2[ind] = dict_h2[side][str(int(parcel))]
                    tex[0].assign(array_h2)                  
                    pits_tex[side] = ana.toAObject(tex)
                    tex_mesh[side] = ana.fusionObjects([meshes[side], pits_tex[side]], method='FusionTexSurfMethod')

                    ref = ana.createReferential()
                    tr = ana.createTransformation([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1] ,ref, ana.centralReferential())
                    tex_mesh[side].assignReferential(ref)
                    tex_mesh[side].setMaterial(front_face='counterclockwise')
                    pits_tex[side].setPalette(colorbar, minVal=INTERVAL[0], maxVal=INTERVAL[1], absoluteMode=True)
                    updateWindow(windows[side], tex_mesh[side])              
                    #paletteViewer.ShowHidePaletteCallback().doit(anatomist.api.cpp.set_AObjectPtr([pits_tex[side].getInternalRep()]))
                    ana.execute('WindowConfig', windows=[windows[side]], cursor_visibility=0)
                    if side == "L":
                        view_quaternions = {'extern' : [0.5, 0.5, 0.5, 0.5],
                                            'intern' : [0.5, -0.5, -0.5, 0.5],
                                            'bottom' : [1, 0, 0, 0],
                                            'top': [0, 0, 1, 0]}
                    else:
                        view_quaternions = {'extern' : [0.5, 0.5, 0.5, 0.5],
                                            'intern' : [0.5, -0.5, -0.5, 0.5],
                                            'bottom' : [1, 0, 0, 0],
                                            'top': [0, 0, 1, 0]}

                    for sd in view_quaternions.keys():
                        q = aims.Quaternion(view_quaternions[sd])
                        windows[side].camera(view_quaternion=view_quaternions[sd],
                                             zoom=0.65)

                        ana.execute('WindowConfig', windows=[windows[side]], snapshot=OUTPUT+'snapshot_bonf'+str(BONF)+'_herit_'+side+'_'+sd+'.png')
                    ana.releaseObject(pits_tex[side])
                    ana.releaseObject(tex_mesh[side])
                    ana.releaseObject(meshes[side])
                    pits_tex[side] = None
                    tex_mesh[side] = None
                    meshes[side] = None
                    #paletteViewer.ShowHidePaletteCallback().doit(anatomist.api.cpp.set_AObjectPtr([pits_tex[side].getInternalRep()]))
                
                # Displaying -log10 p-values on the template
                for side in sides:
                    template_mesh  = '/neurospin/brainomics/folder_gii/'+side.lower()+'h.inflated.white.gii'
                    meshes[side] = ana.loadObject(template_mesh)
                    windows[side] = ana.createWindow('3D', geometry=[0, 0, 584, 584])
                    file_parcels_on_atlas = os.path.join(group_path, side+'.fsaverage164k.label.gii')
                    array_parcels = gio.read(file_parcels_on_atlas).darrays[0].data
                    array_pval = np.zeros(len(array_parcels))
                    parcels_org =  np.unique(array_parcels)
                    parcels_org = parcels_org[1:]
                    for parcel in parcels_org:
                        if dict_pval[side].has_key(str(int(parcel))):
                            if dict_pval[side][str(int(parcel))] < BONFTHRESHOLD:
                                ind = np.where(array_parcels == parcel)[0]
                                #if dict_pval[side][str(int(parcel))] < 0.05/1000:
                                array_pval[ind] = -np.log10(dict_pval[side][str(int(parcel))])
                    tex[0].assign(array_pval)                  
                    pits_tex[side] = ana.toAObject(tex)
                    tex_mesh[side] = ana.fusionObjects([meshes[side], pits_tex[side]], method='FusionTexSurfMethod')

                    ref = ana.createReferential()
                    tr = ana.createTransformation([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1] ,ref, ana.centralReferential())
                    tex_mesh[side].assignReferential(ref)
                    tex_mesh[side].setMaterial(front_face='counterclockwise')
                    pits_tex[side].setPalette(colorbar2, minVal=INTERVAL2[0], maxVal=INTERVAL2[1], absoluteMode=True)
                    updateWindow(windows[side], tex_mesh[side])              
                    #paletteViewer.ShowHidePaletteCallback().doit(anatomist.api.cpp.set_AObjectPtr([pits_tex[side].getInternalRep()]))
                    ana.execute('WindowConfig', windows=[windows[side]], cursor_visibility=0)
                    if side == "L":
                        view_quaternions = {'intern' : [0.5, 0.5, 0.5, 0.5],
                                            'extern' : [0.5, -0.5, -0.5, 0.5],
                                            'bottom' : [1, 0, 0, 0],
                                            'top': [0, 0, 1, 0]}
                    else:
                        view_quaternions = {'extern' : [0.5, 0.5, 0.5, 0.5],
                                            'intern' : [0.5, -0.5, -0.5, 0.5],
                                            'bottom' : [1, 0, 0, 0],
                                            'top': [0, 0, 1, 0]}

                    for sd in view_quaternions.keys():
                        q = aims.Quaternion(view_quaternions[sd])
                        windows[side].camera(view_quaternion=view_quaternions[sd],
                                             zoom=0.65)

                        ana.execute('WindowConfig', windows=[windows[side]], snapshot=OUTPUT+'snapshot_bonf'+str(BONF)+'_pval_'+side+'_'+sd+'.png')
                    ana.releaseObject(pits_tex[side])
                    ana.releaseObject(tex_mesh[side])
                    ana.releaseObject(meshes[side])
                    pits_tex[side] = None
                    tex_mesh[side] = None
                    meshes[side] = None
                    #paletteViewer.ShowHidePaletteCallback().doit(anatomist.api.cpp.set_AObjectPtr([pits_tex[side].getInternalRep()]))


