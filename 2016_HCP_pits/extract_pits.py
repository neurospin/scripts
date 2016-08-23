"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re, json, time
from sklearn.cluster import KMeans
from sklearn import preprocessing
import nibabel.gifti.giftiio as gio

import anatomist.api 
ana = anatomist.api.Anatomist()
import paletteViewer
from soma import aims

def updateWindow(window, obj):
    window.addObjects(obj)
    obj.setChanged()
    obj.notifyObservers()

## INPUTS ###
path = '/media/yl247234/SAMSUNG/HCP_from_cluster/databaseBV/hcp/'
temp_file_s_ids='/home/yl247234/s_ids_lists/s_ids.json'
sides = ['R', 'L']
windows = {}
white_meshes = {}
tex_pits = {}
tex_mesh_pits = {}
## CONSTANTS ##
FILTERING = True
# IF YOU WANT TO SCALE THE DATA IN THE WAY THAT DPF PLAYS AS IMPORTANT ROLE AS POSITION SET BOOLEAN TO True #
SCALING_DATA = False
THRESHOLD_DPF = 1.3
NB_CLUSTERS = 50

with open(temp_file_s_ids, 'r') as f:
    data = json.load(f)


## K-means applied to cluster all the pits ##
for side in sides:
    s_ids  = list(json.loads(data))
    template_mesh  = '/neurospin/imagen/workspace/cati/templates/average_'+side+'mesh_BL.gii'
    mesh_average = gio.read(template_mesh)
    vertices_origin = mesh_average.darrays[0].data
    triangles_origin = mesh_average.darrays[1].data

    X = np.array([])
    t = time.time()
    pits_data = np.array([])
    full_indexes = np.array([])
    pits_nb = []
    s_ids_update = []
    for s_id in s_ids:
        file_pits_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                          "default_analysis", "segmentation", "mesh",
                                          "surface_analysis", ""+s_id+"_"+side+"white_pits_on_atlas.gii")
        file_DPF_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                          "default_analysis", "segmentation", "mesh",
                                          "surface_analysis", ""+s_id+"_"+side+"white_DPF_on_atlas.gii")
        if os.path.isfile(file_pits_on_atlas) and os.path.isfile(file_DPF_on_atlas):
            s_ids_update.append(s_id)
            array_pits = gio.read(file_pits_on_atlas).darrays[0].data
            array_DPF = gio.read(file_DPF_on_atlas).darrays[0].data
            if FILTERING:
                array_pits = array_pits*array_DPF>THRESHOLD_DPF
                array_pits = array_pits.astype(int)
            
            index_pits = np.nonzero(array_pits)[0]
            pits_nb.append(len(index_pits))
            temp = np.zeros((len(index_pits),4))
            # Here I add an aribitrary multiply by 3 in order to increase DPF variations importance a bit
            temp += np.concatenate((vertices_origin[index_pits], 9*array_DPF[index_pits].reshape(len(index_pits),1)), axis=1)
            if X.size == 0:
                X = temp
                full_indexes = index_pits
            else:
                X = np.concatenate((X,temp))
                full_indexes = np.concatenate((full_indexes, index_pits))
        else:
            print "WARNING " +s_id+" for side "+side+" doesn't exist !"
    if SCALING_DATA:
        X = preprocessing.scale(X)
    estimator = KMeans(n_clusters=NB_CLUSTERS, init='random', n_init=100, max_iter=1000, tol=0.001, n_jobs = -2, verbose=0)
    estimator.fit(X)
    print "Elapsed time for Kmeans : "+ str(time.time()-t)
    nb_index = len(full_indexes)
    vertexs = np.hstack((full_indexes.reshape(nb_index,1),estimator.labels_.reshape(nb_index,1)+1))
    pits_data = array_pits
    for vert in vertexs:
        pits_data[vert[0]] = vert[1]

    white_meshes[side] = ana.loadObject(template_mesh)
    windows[side] = ana.createWindow('3D')
    aims_pits = aims.TimeTexture(dtype='FLOAT')
    aims_pits[0].assign(pits_data)
    tex_pits[side] = ana.toAObject(aims_pits)
    tex_mesh_pits[side] = ana.fusionObjects([white_meshes[side], tex_pits[side]], method='FusionTexSurfMethod')
    tex_pits[side].setPalette('zfun-EosB')
    updateWindow(windows[side], tex_mesh_pits[side])
    ana.execute('WindowConfig', windows=[windows[side]], cursor_visibility=0)
    
    s_ids = s_ids_update
    DATA_DPF = np.zeros((len(s_ids), NB_CLUSTERS))*np.nan
    for k,s_id  in enumerate(s_ids):
        ind_bottom = sum(pits_nb[:k])
        """file_pits_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                          "default_analysis", "segmentation", "mesh",
                                          "surface_analysis", ""+s_id+"_"+side+"white_pits_on_atlas.gii")
        file_DPF_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                          "default_analysis", "segmentation", "mesh",
                                          "surface_analysis", ""+s_id+"_"+side+"white_DPF_on_atlas.gii")
        if os.path.isfile(file_pits_on_atlas) and os.path.isfile(file_DPF_on_atlas):
            array_pits = gio.read(file_pits_on_atlas).darrays[0].data
            array_DPF = gio.read(file_DPF_on_atlas).darrays[0].data"""
        
        for i in range(pits_nb[k]):
            clust = estimator.labels_[i+ind_bottom]
            if np.isnan(DATA_DPF[k, clust]):
                DATA_DPF[k, clust] = X[i+ind_bottom, 3]
            elif DATA_DPF[k, clust] < X[i+ind_bottom, 3]:
                DATA_DPF[k, clust] = X[i+ind_bottom, 3]


    DATA_DPF = np.nan_to_num(DATA_DPF)
    index_columns_kept = []
    for j in range(DATA_DPF.shape[1]):
        print np.count_nonzero(DATA_DPF[:,j])
        if np.count_nonzero(DATA_DPF[:,j]) > DATA_DPF.shape[0]*0.5:
            index_columns_kept.append(j)
    print index_columns_kept
    OUTPUT = '/neurospin/brainomics/2016_HCP/pheno_DPF_pits_bis/'
    for index in index_columns_kept:
        df3 = pd.DataFrame()
        df3['IID'] = np.asarray(s_ids)[np.nonzero(DATA_DPF[:,index])].tolist()
        df3['Pit_'+str(index)] = DATA_DPF[:,index][np.nonzero(DATA_DPF[:,index])]
        if not os.path.exists(OUTPUT):
            os.makedirs(OUTPUT)
        output = OUTPUT+'test_DPF_pit'+str(index)+"side"+side
        df3.to_csv(output+'.csv',  header=True, index=False)
    

    df = pd.DataFrame()
    for j in range(DATA_DPF.shape[1]):
        df['Pit_'+str(j)] = DATA_DPF[:,j]
    df[df != 0] = 2
    df[df == 0] = 1
    df['IID'] = np.asarray(s_ids)
    OUTPUT2 = OUTPUT + 'case_control/'
    if not os.path.exists(OUTPUT2):
        os.makedirs(OUTPUT2)
    output = OUTPUT2+'test_DPF_all_pits_side'+side
    df.to_csv(output+'.csv', sep= ',',  header=True, index=False)
