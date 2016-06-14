"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re
import gdist
import nibabel.gifti.giftiio as gio

## INPUTS ###
#right 'R' or left 'L'
side = 'L'
sides = {'R': 'Right',
         'L': 'Left'}
INPUT = "/neurospin/brainomics/2016_sulcal_pits/extracting_pits_V1/"

## OUTPUT ##
OUTPUT = "/neurospin/brainomics/2016_sulcal_pits/pheno_pits/test0_V1/"

path = INPUT+sides[side]+'/kept/pits/'
s_ids = []
for filename in glob.glob(os.path.join(path,'*.txt')):
    m = re.search(path+side+'(.+?)_pits_kept.txt', filename)
    if m:
        label = m.group(1)
        if '000' in label:
            s_ids.append(label)

filename_average = '/neurospin/imagen/workspace/cati/BVdatabase/average_'+side+'mesh_BL.gii'
mesh_average = gio.read(filename_average)
vertices_origin = mesh_average.darrays[0].data
triangles_origin = mesh_average.darrays[1].data
file_parcels_marsAtlas = '/neurospin/imagen/workspace/cati/hiphop138-Template/hiphop138_'+side+'white_parcels_marsAtlas.gii'
parcels_data=  gio.read(file_parcels_marsAtlas).darrays[0].data
"""cortex = np.nonzero(parcels_data)[0] # because from my understanding the medial wall (which is the non-cortex part) is the parcel of value 0
keep = np.zeros(triangles_origin.shape[0])
for i in [0, 1, 2]:
    keep += np.array([item in cortex for item in triangles_origin[:, i]])
ind = np.where(keep == 3)[0]
triangles = np.array(triangles_origin[ind], dtype=np.int32)
triangles_old = np.array(triangles_origin[ind], dtype=np.int32)
for c, i in enumerate(cortex):
    triangles[np.where(triangles_old == i)] = c
vertices = vertices_origin[cortex].astype(np.float64)
triangles = triangles.astype(np.int32)
"""

"""# probably gonna be to slow even if we restrain ourself to the parcel
def dijkstra_on_mesh(vertices, triangles, s, d):
    length = np.zeros(len(vertices))
    for i in range(len(vertices)):
        length(i) = np.inf
    l(s) = 0
    Set = set([])

    while d not in Set:
        for j"""
    
    




parcels = {}
parcels_init = {}
keys_parcels = np.loadtxt(INPUT+sides[side]+'_parcels_kept.txt')
s_ids_init = np.loadtxt(INPUT+sides[side]+'_s_ids_init.txt')
s_ids_init  = ['%012d' % int(i) for i in s_ids_init]
init_pits = np.loadtxt(INPUT+sides[side]+'_init_pits.txt')
# Building our tables for comparison
for j, parcel in enumerate(keys_parcels):
    s_id = s_ids_init[j]
    pits = np.loadtxt(INPUT+sides[side]+'/kept/pits/'+side+s_id+'_pits_kept.txt')
    parcels_numbers = np.loadtxt(INPUT+sides[side]+'/kept/parcels/'+side+s_id+'_parcels_numbers_kept.txt')
    index = np.where(parcels_numbers==parcel)[0].tolist()
    parcels_init[str(parcel)] = init_pits[j]
    for i, ind in enumerate(index):
        parcels["Parcel"+str(parcel)+"_pit"+str(i)] = [pits[ind]] 


DATA_DPF = np.zeros((len(s_ids), sum(init_pits),2))
for k,s_id in enumerate(s_ids):
    parcel_previous = np.nan
    #pits_index = np.loadtxt(INPUT+sides[side]+'/kept/pits/'+side+s_id+'_pits_kept.txt')
    parcels_number = np.loadtxt(INPUT+sides[side]+'/kept/parcels/'+side+s_id+'_parcels_numbers_kept.txt')
    DPF = np.loadtxt(INPUT+sides[side]+'/kept/DPF/'+side+s_id+'_DPF_pits_kept.txt')
    pits = np.loadtxt(INPUT+sides[side]+'/kept/pits/'+side+s_id+'_pits_kept.txt')
    print s_id
    for j, parcel in enumerate(parcels_number):
        if parcel_previous == parcel:
            pass
        else:
            """parcel_zone = np.where(parcels_data == parcel)
            keep = np.zeros(triangles_origin.shape[0])
            for l in [0, 1, 2]:
                keep += np.array([item in parcel_zone[0] for item in triangles_origin[:, l]])
            ind = np.where(keep == 3)[0]
            triangles = np.array(triangles_origin[ind], dtype=np.int32)
            triangles_old = np.array(triangles_origin[ind], dtype=np.int32)
            for c, l in enumerate(parcel_zone):
                triangles[np.where(triangles_old == l)] = c
                vertices = vertices_origin[parcel_zone].astype(np.float64)
                triangles = triangles.astype(np.int32)"""
            parcel_previous = parcel
            index_start_parcel = sum(init_pits[:np.where(keys_parcels == parcel)[0]])
        min_dist_pit = np.inf
        pit = np.nan

        #current_pit_index_update = np.where(parcel_zone[0] == pits[j])        
        for i in range(np.int(parcels_init[str(parcel)])):
            #dist = np.square(lat[j]-parcels["Parcel"+str(parcel)+"_pit"+str(i)][0])+np.square(lon[j]-parcels["Parcel"+str(parcel)+"_pit"+str(i)][1])
            #target_pit_index_update = np.where(parcel_zone[0] == parcels["Parcel"+str(parcel)+"_pit"+str(i)][0])
            #dist = gdist.compute_gdist(vertices.astype(np.float64), triangles.astype(np.int32), source_indices=np.asarray(current_pit_index_update[0], dtype=np.int32), target_indices=np.asarray(target_pit_index_update[0], dtype=np.int32))[0]
            dist = np.sum(np.square(vertices_origin[parcels["Parcel"+str(parcel)+"_pit"+str(i)][0]]- vertices_origin[pits[j]]))
            if dist < min_dist_pit:
                min_dist_pit = dist
                pit = i
        # case when DPF is really 0 is ignored for th moment
        if not (DATA_DPF[k,index_start_parcel+pit,0] == 0):
            # Here should had a comparison with the previously calculated distance
            #print "Warning you are about to replace an existing DPF value ! " + str(DATA_DPF[k,index_start_parcel+pit])+ " by " + str([DPF[j], dist])
            if DATA_DPF[k,index_start_parcel+pit,1] > dist:
                DATA_DPF[k,index_start_parcel+pit,:] = [DPF[j], dist]
            else:
                #print "Value not replaced."
                pass
        else:
            DATA_DPF[k,index_start_parcel+pit,:] = [DPF[j], dist]
        
        
index_columns_kept = []
for j in range(DATA_DPF.shape[1]):
    print np.count_nonzero(DATA_DPF[:,j,0]) 
    if np.count_nonzero(DATA_DPF[:,j,0]) > DATA_DPF.shape[0]-400:
        index_columns_kept.append(j)


for index in index_columns_kept:
    df = pd.DataFrame()
    df['IID'] = np.asarray(s_ids)[np.nonzero(DATA_DPF[:,index,0])].tolist()
    df['FID'] = df['IID']
    df.index = df['IID']
    df['Pit_'+str(index)] = DATA_DPF[:,index,0][np.nonzero(DATA_DPF[:,index,0])]
    df.to_csv(OUTPUT+sides[side]+'_Pit_'+str(index)+'.phe', sep= '\t',  header=True, index=False)
