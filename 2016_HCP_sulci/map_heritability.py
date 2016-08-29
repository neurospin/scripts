"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re, json
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
features = ['GM_thickness', 'hull_junction_length_native', 'maxdepth_native', 'meandepth_native', 'opening', 'surface_native']

# 62 sulci 6 features
Bonferroni_correction = 62*6

sides  = ['R', 'L']
windows = {}
sulci = {}

file_pval = '/neurospin/brainomics/2016_HCP/sulci_dict_pval.json'
with open(file_pval, 'r') as f:
    data = json.load(f)
sulci_dict_pval = json.loads(data)

file_h2 = '/neurospin/brainomics/2016_HCP/sulci_dict_h2.json'
with open(file_h2, 'r') as f:
    data = json.load(f)
sulci_dict_h2 = json.loads(data)

for feature in features:
    for side in sides:
        file_graph = '/neurospin/brainomics/2016_HCP/spam/'+side+'spam_model_meshes_1.arg'
        sulci[side] = ana.loadObject(file_graph)
        sulci[side].setColorMode(sulci[side].PropertyMap)
        graph = ana.toAimsObject(sulci[side])
        for vertex in graph.vertices():
            if vertex.has_key('name'):
                sulcus = vertex['name']
                print sulcus
                if sulci_dict_h2.has_key(sulcus):
                    if sulci_dict_h2[sulcus].has_key(feature) and sulci_dict_pval[sulcus][feature] < 0.05/Bonferroni_correction:
                        vertex['heritability'] = sulci_dict_h2[sulcus][feature]
                    else:
                        vertex['heritability'] = 0
                else:
                    vertex['heritability'] = 0
        sulci[side].setColorProperty('heritability')
        sulci[side].setPalette(colorbar, minVal=INTERVAL[0], maxVal=INTERVAL[1], absoluteMode=True)
        windows[side] = ana.createWindow('3D', geometry=[0, 0, 584, 584])
        updateWindow(windows[side], sulci[side])
        if side == "R":
            view_quaternions = {'intern' : [0.5, 0.5, 0.5, 0.5],
                                'extern' : [0.5, -0.5, -0.5, 0.5]}
        else:
            view_quaternions = {'extern' : [0.5, 0.5, 0.5, 0.5],
                                'intern' : [0.5, -0.5, -0.5, 0.5]}
        for sd in view_quaternions.keys():
            q = aims.Quaternion(view_quaternions[sd])
            """bbox = [100, 1300, 50]
            bbox_m = [-100, -70, -120]
            bbox_t = [q.transform([bbox[0], 0, 0])]
            bbox_t.append(q.transform([bbox_m[0], 0, 0]))
            bbox_t.append(q.transform([0, bbox[1], 0]))
            bbox_t.append(q.transform([0, bbox_m[1], 0]))
            bbox_t.append(q.transform([0, 0, bbox[2]]))
            bbox_t.append(q.transform([0, 0, bbox_m[2]]))
            bbox_min = (min([b[0] for b in bbox_t]),
                        min([b[1] for b in bbox_t]),
                        min([b[2] for b in bbox_t]))
            bbox_max = (max([b[0] for b in bbox_t]),
                        max([b[1] for b in bbox_t]),
                        max([b[2] for b in bbox_t]))"""
            windows[side].camera(view_quaternion=view_quaternions[sd],
                                 zoom=1)
            #, boundingbox_min=bbox_min,
            #boundingbox_max=bbox_max)
            ana.execute('WindowConfig', windows=[windows[side]], snapshot='/home/yl247234/Images/auto_snap/snapshot_'+feature+'_'+side+'_'+sd+'.jpg')
