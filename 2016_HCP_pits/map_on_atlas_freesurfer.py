"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import json, os
import numpy as np
import nibabel.gifti.giftiio as gio



path = '/media/yl247234/SAMSUNG/HCP_from_cluster/Freesurfer_mesh_database/hcp'
sides = ['R', 'L']
windows = {}
white_meshes = {}
tex_pits = {}
tex_mesh_pits = {}
temp_file_s_ids='/home/yl247234/s_ids_lists/s_ids.json'
with open(temp_file_s_ids, 'r') as f:
    data = json.load(f)
s_ids  = list(json.loads(data))



avg_sum = {}
for side in sides:
    avg_sum[side] = 0
    count = 0
    template_mesh = '/neurospin/brainomics/folder_gii/'+side.lower()+'h.inflated.white.gii'
    for s_id in s_ids[:]:
        file_white_mesh = os.path.join(path, s_id, "t1mri", "BL",
                                       "default_analysis", "segmentation", "mesh",
                                       ""+ s_id+ "_"+side+"white.gii")
        file_pits_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                          "default_analysis", "segmentation", "mesh",
                                          "surface_analysis", ""+s_id+"_"+side+"white_pits_on_atlas.gii")
        file_DPF_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                         "default_analysis", "segmentation", "mesh",
                                         "surface_analysis", ""+s_id+"_"+side+"white_DPF_on_atlas.gii")
        file_pits = os.path.join(path, s_id, "t1mri", "BL",
                                          "default_analysis", "segmentation", "mesh",
                                          "surface_analysis", ""+s_id+"_"+side+"white_pits.gii")
        file_DPF = os.path.join(path, s_id, "t1mri", "BL",
                                         "default_analysis", "segmentation", "mesh",
                                         "surface_analysis", ""+s_id+"_"+side+"white_DPF.gii")
        if os.path.isfile(file_pits_on_atlas) and os.path.isfile(file_DPF_on_atlas):
            count +=1
            array_pits_on_atlas = gio.read(file_pits_on_atlas).darrays[0].data
            array_DPF_on_atlas = gio.read(file_DPF_on_atlas).darrays[0].data
            array_pits = gio.read(file_pits).darrays[0].data
            array_DPF = gio.read(file_DPF).darrays[0].data
            pits = np.nonzero(array_pits)
            print "NUMBER PITS: " +str(len(pits[0]))
            """print pits[0]
            print array_DPF[pits]
            print len(array_DPF[pits])"""

            pits_on_atlas = np.nonzero(array_pits_on_atlas)
            print "NUMBER PITS ON ATLAS: " +str(len(pits_on_atlas[0]))
        avg_sum[side] += float(len(pits_on_atlas[0]))/len(pits[0])
    avg_sum[side] = avg_sum[side]/count

print avg_sum







"""
from soma import aims

import anatomist.api 
ana = anatomist.api.Anatomist()

def updateWindow(window, obj):
    window.addObjects(obj)
    obj.setChanged()
    obj.notifyObservers()
file_pits_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                  "default_analysis", "segmentation", "mesh",
                                  "surface_analysis", ""+s_id+"_"+side+"white_pits_on_atlas.gii")
white_meshes[side] = ana.loadObject(template_mesh)
windows[side] = ana.createWindow('3D')
tex_pits[side] = ana.loadObject(pits)
tex_mesh_pits[side] = ana.fusionObjects([white_meshes[side], tex_pits[side]], method='FusionTexSurfMethod')
tex_pits[side].setPalette('actif_ret')
ref = ana.createReferential()
tr = ana.createTransformation([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1] ,ref, ana.centralReferential())
tex_mesh_pits[side].assignReferential(ref)
tex_mesh_pits[side].setMaterial(front_face='counterclockwise')
updateWindow(windows[side], tex_mesh_pits[side])
ana.execute('WindowConfig', windows=[windows[side]], cursor_visibility=0)
"""
