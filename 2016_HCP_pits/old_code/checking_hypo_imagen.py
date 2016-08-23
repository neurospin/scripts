import json, os
import numpy as np
import nibabel.gifti.giftiio as gio

import anatomist.api 
ana = anatomist.api.Anatomist()
import paletteViewer

def updateWindow(window, obj):
    window.addObjects(obj)
    obj.setChanged()
    obj.notifyObservers()

sides = ['R', 'L']
side = sides[0]
template_mesh  = '/neurospin/imagen/workspace/cati/templates/average_'+side+'mesh_BL.gii'

count = 0
avg_sum = 0
path0 = '/neurospin/imagen/workspace/cati/BVpitsdatabase/'
centers = os.listdir(path0)

for center in centers:
    path = path0+center+'/'

    s_ids = os.listdir(path)
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
            """print pits_on_atlas[0]
            print array_DPF_on_atlas[pits_on_atlas]
            print len(array_DPF_on_atlas[pits_on_atlas])"""

            """white_mesh = ana.loadObject(file_white_mesh)
            window = ana.createWindow('3D')
            tex_pits = ana.loadObject(file_pits)
            tex_mesh_pits = ana.fusionObjects([white_mesh, tex_pits], method='FusionTexSurfMethod')
            tex_pits.setPalette('actif_ret')
            updateWindow(window, tex_mesh_pits)
            white_mesh2 = ana.loadObject(template_mesh)
            window2 = ana.createWindow('3D')
            tex_pits_on_atlas = ana.loadObject(file_pits_on_atlas)
            tex_mesh_pits_on_atlas = ana.fusionObjects([white_mesh2, tex_pits_on_atlas], method='FusionTexSurfMethod')
            tex_pits_on_atlas.setPalette('actif_ret')
            updateWindow(window2, tex_mesh_pits_on_atlas)
            """
            avg_sum += float(len(pits_on_atlas[0]))/len(pits[0])
print avg_sum/count
