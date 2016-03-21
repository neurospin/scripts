"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re

import nibabel.gifti.giftiio as gio

## INPUTS ###
#right 'R' or left 'L'
side = 'L'
sides = {'R': 'Right',
         'L': 'Left'}
path = '/neurospin/imagen/workspace/cati/BVdatabase/'
file_parcels_marsAtlas = path+side+'_clusters_default_parameters2.gii'
parcels_data=  gio.read(file_parcels_marsAtlas).darrays[0].data
NUMBER_PARCELS = max(parcels_data)

folder = "extracting_pits_V4/"
INPUT = "/neurospin/brainomics/2016_sulcal_pits/"+folder

path = INPUT+sides[side]+'/all/pits/'
s_ids = []
for filename in glob.glob(os.path.join(path,'*.txt')):
    m = re.search(path+side+'(.+?)_pits.txt', filename)
    if m:
        label = m.group(1)
        if '000' in label:
            s_ids.append(label)

df = pd.DataFrame()
df['index'] = range(NUMBER_PARCELS)
df.index = df['index']

for k,s_id in enumerate(s_ids):
    parcels_numbers = np.loadtxt(INPUT+sides[side]+'/all/parcels/'+side+s_id+'_parcels_numbers.txt')
    pits_per_parcels = [len(parcels_numbers)-np.count_nonzero(parcels_numbers-p) for p in range(NUMBER_PARCELS)]
    df[s_id] = pits_per_parcels

df = df[s_ids]

df.to_csv(INPUT+sides[side]+'_statistic_pits.csv', sep= '\t',  header=True, index=False)


count = 0
parcels_kept = []
for i in range(NUMBER_PARCELS):
    if np.mean(df.loc[i]) > 0.2 and np.std(df.loc[i]) < 3:
        count += 1
        print "\n"
        print "Parcel number: " + str(i)
        parcels_kept.append(i)
        print "Mean: " + str(np.mean(df.loc[i]))
        print "Std: " + str(np.std(df.loc[i]))

s_ids_max = []
nb_pits_max = []
for p in parcels_kept:
    # I know several subjects can reach the maximum number of pits in a parcel for the moment we just take the one given by argmax (maybe to be changed later)
    s_ids_max.append(np.argmax(df.loc[p]))
    nb_pits_max.append(np.amax(df.loc[p]))

path_saved = INPUT
filename_sav = sides[side]+'_s_ids_max.txt'
thefile = open(path_saved+ filename_sav, 'w')
for item in s_ids_max:
    #print item
    thefile.write("%s\n" % item)
thefile.close()

filename_sav = sides[side]+'_parcels_kept.txt'
filename_sav2 = sides[side]+'_max_pits.txt'
np.savetxt(path_saved+filename_sav, parcels_kept)
np.savetxt(path_saved+filename_sav2, nb_pits_max)

# Subjects used to initialize 

s_ids_init = []
nb_pits_init = []
for i,p in enumerate(parcels_kept):
    nb_pits_init.append(round(np.mean(df.loc[p])+0.5))
    index_s_id = np.random.choice(np.where(df.loc[p]== nb_pits_init[i])[0])
    s_ids_init.append(s_ids[index_s_id])
    
filename_sav = sides[side]+'_s_ids_init.txt'
thefile = open(path_saved+ filename_sav, 'w')
for item in s_ids_init:
    #print item
    thefile.write("%s\n" % item)
thefile.close()

filename_sav = sides[side]+'_parcels_kept.txt'
filename_sav2 = sides[side]+'_init_pits.txt'
np.savetxt(path_saved+filename_sav, parcels_kept)
np.savetxt(path_saved+filename_sav2, nb_pits_init)



import itertools
for s_id in s_ids:
    pits_index = np.loadtxt(INPUT+sides[side]+'/all/pits/'+side+s_id+'_pits.txt')
    parcels_numbers = np.loadtxt(INPUT+sides[side]+'/all/parcels/'+side+s_id+'_parcels_numbers.txt')
    DPF_pits = np.loadtxt(INPUT+sides[side]+'/all/DPF/'+side+s_id+'_DPF_pits.txt')
    index_parcels = [np.where(parcels_numbers==p)[0].tolist() for p in parcels_kept]
    index_parcels = list(itertools.chain.from_iterable(index_parcels))
    pits_index_kept = pits_index[index_parcels]
    parcels_numbers_kept = parcels_numbers[index_parcels]
    DPF_pits_kept = DPF_pits[index_parcels]

    np.savetxt(INPUT+sides[side]+'/kept/pits/'+side+s_id+'_pits_kept.txt', pits_index_kept)
    np.savetxt(INPUT+sides[side]+'/kept/parcels/'+side+s_id+'_parcels_numbers_kept.txt', parcels_numbers_kept)
    np.savetxt(INPUT+sides[side]+'/kept/DPF/'+side+s_id+'_DPF_pits_kept.txt', DPF_pits_kept)
