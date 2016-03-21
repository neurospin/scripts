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
INPUT = "/neurospin/brainomics/2016_sulcal_pits/extracting_pits_V4/"

## OUTPUT ##
OUTPUT = "/neurospin/brainomics/2016_sulcal_pits/pheno_pits/extract_v4/test1/"
OUTPUT2 = OUTPUT+"pits_numeros/"
if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)
    os.makedirs(OUTPUT+'binary_analysis/')
    os.makedirs(OUTPUT2)
path_s = os.path.join(OUTPUT,sides[side])
if not os.path.exists(path_s):
    os.makedirs(path_s)
        
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
file_parcels_marsAtlas = '/neurospin/imagen/workspace/cati/BVdatabase/'+side+'_clusters_default_parameters.gii'
parcels_data=  gio.read(file_parcels_marsAtlas).darrays[0].data


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


DATA_DPF = np.zeros((len(s_ids), sum(init_pits),3))
for k,s_id in enumerate(s_ids):
    parcel_previous = np.nan
    #pits_index = np.loadtxt(INPUT+sides[side]+'/kept/pits/'+side+s_id+'_pits_kept.txt')
    parcels_number = np.loadtxt(INPUT+sides[side]+'/kept/parcels/'+side+s_id+'_parcels_numbers_kept.txt')
    DPF = np.loadtxt(INPUT+sides[side]+'/kept/DPF/'+side+s_id+'_DPF_pits_kept.txt')
    pits = np.loadtxt(INPUT+sides[side]+'/kept/pits/'+side+s_id+'_pits_kept.txt')
    print s_id
    for j, parcel in enumerate(parcels_number):
        index_start_parcel = sum(init_pits[:np.where(keys_parcels == parcel)[0]])
        min_dist_pit = np.inf
        pit = np.nan
        for i in range(np.int(parcels_init[str(parcel)])):
            #dist = np.square(lat[j]-parcels["Parcel"+str(parcel)+"_pit"+str(i)][0])+np.square(lon[j]-parcels["Parcel"+str(parcel)+"_pit"+str(i)][1])
            #dist = gdist.compute_gdist(vertices, triangles, source_indices=np.asarray([pits[j]], dtype=np.int32), target_indices=np.asarray(parcels["Parcel"+str(parcel)+"_pit"+str(i)], dtype=np.int32))[0]
            dist = np.sum(np.square(vertices_origin[parcels["Parcel"+str(parcel)+"_pit"+str(i)][0]]- vertices_origin[pits[j]]))
            if dist < min_dist_pit:
                min_dist_pit = dist
                pit = i
        # case when DPF is really 0 is ignored for th moment
        if not (DATA_DPF[k,index_start_parcel+pit,0] == 0):
            # Here should had a comparison with the previously calculated distance
            #print "Warning you are about to replace an existing DPF value ! " + str(DATA_DPF[k,index_start_parcel+pit])+ " by " + str([DPF[j], dist])
            if DATA_DPF[k,index_start_parcel+pit,1] > dist:
                DATA_DPF[k,index_start_parcel+pit,:] = [DPF[j], dist, pits[j]]
            else:
                #print "Value not replaced."
                pass
        else:
            DATA_DPF[k,index_start_parcel+pit,:] = [DPF[j], dist, pits[j]]
        
        
index_columns_kept = []
for j in range(DATA_DPF.shape[1]):
    print np.count_nonzero(DATA_DPF[:,j,0]) 
    if np.count_nonzero(DATA_DPF[:,j,0]) > DATA_DPF.shape[0]-800:
        index_columns_kept.append(j)



for index in index_columns_kept:
    df = pd.DataFrame()
    df['IID'] = np.asarray(s_ids)[np.nonzero(DATA_DPF[:,index,0])].tolist()
    df['FID'] = df['IID']
    df.index = df['IID']
    df2 = df
    df['Pit_'+str(index)] = DATA_DPF[:,index,0][np.nonzero(DATA_DPF[:,index,0])]
    df.to_csv(OUTPUT+sides[side]+'/'+sides[side]+'_Pit_'+str(index)+'.phe', sep= '\t',  header=True, index=False)
    df2['Pit_numero'] = DATA_DPF[:,index,2][np.nonzero(DATA_DPF[:,index,0])]
    df2.to_csv(OUTPUT2+sides[side]+'_Pit'+str(index)+'_positions.csv', sep= '\t',  header=True, index=False)

df = pd.DataFrame()
df2 = pd.DataFrame()
for j in range(DATA_DPF.shape[1]):
    df['Pit_'+str(j)] = DATA_DPF[:,j,0]
    df2['Pit_numero'+str(j)] = DATA_DPF[:,j,2]
df[df != 0] = 2
df[df == 0] = 1

df['IID'] = np.asarray(s_ids)
df['FID'] = df['IID']
df.index = df['IID']
columns = df.columns-['IID','FID']
columns_f =['IID','FID']+columns.tolist()
df = df[columns_f]
df2['IID'] = np.asarray(s_ids)
df2['FID'] = df2['IID']
df2.index = df2['IID']
columns = df2.columns-['IID','FID']
columns_f =['IID','FID']+columns.tolist()
df2 = df2[columns_f]


df.to_csv(OUTPUT+'binary_analysis/'+sides[side]+'_Pits_binary.phe', sep= '\t',  header=True, index=False)
df2.to_csv(OUTPUT2+sides[side]+'_Pits_binary_positions.csv', sep= '\t',  header=True, index=False)
