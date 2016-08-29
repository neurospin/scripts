"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re, json
import nibabel.gifti.giftiio as gio

database = 'Freesurfer_mesh_database'
#database = 'databaseBV'
cohort = 'hcp'
OUTPUT = '/media/yl247234/SAMSUNG/'+cohort+'/'+database+'/pits_density/'
path = '/media/yl247234/SAMSUNG/'+cohort+'/'+database+'/hcp/'
temp_file_s_ids='/home/yl247234/s_ids_lists/s_ids.json'
with open(temp_file_s_ids, 'r') as f:
    data = json.load(f)
s_ids  = list(json.loads(data))

filename_R_gii = 'R_average_pits_smoothed0.7_60.gii'
filename_L_gii = 'L_average_pits_smoothed0.7_60.gii'
filename_L_R_gii = 'total_average_pits_smoothed0.7_60_sym.gii'

pits_data_R = np.array([])
pits_data_L = np.array([])
count_R = 0
count_L = 0
total_density = np.array([])
for s_id in s_ids:
    path_s =  path+s_id+'/'
    if "imagen" in path and "Freesurfer" not in path:
        s_id = s_id[len(s_id)-12:]
    file_pits_R = path_s + 't1mri/BL/default_analysis/segmentation/mesh/surface_analysis_sym/'+s_id+'_Rwhite_pits_smoothed0.7_60_on_atlas.gii'
    file_pits_L = path_s + 't1mri/BL/default_analysis/segmentation/mesh/surface_analysis_sym/'+s_id+'_Lwhite_pits_smoothed0.7_60_on_atlas.gii'
    if os.path.isfile(file_pits_R):
        count_R +=1
        if pits_data_R.size == 0:
            pits_data_R = gio.read(file_pits_R).darrays[0].data
        else:
            pits_data_R += gio.read(file_pits_R).darrays[0].data
    if os.path.isfile(file_pits_L):
        count_L +=1       
        if pits_data_L.size == 0:
            pits_data_L = gio.read(file_pits_L).darrays[0].data
        else:
            pits_data_L += gio.read(file_pits_L).darrays[0].data

pits_data_R_temp = pits_data_R/count_R
pits_data_L_temp = pits_data_L/count_L

total_density = (pits_data_R_temp+pits_data_L_temp)/2

"""g_R = gio.read(file_pits_R)
g_L = gio.read(file_pits_L)
g_R.darrays[0].data = pits_data_R_temp
g_L.darrays[0].data = pits_data_L_temp
gio.write(g_R, OUTPUT+filename_R_gii)
gio.write(g_L, OUTPUT+filename_L_gii)"""

g_L_R = gio.read(file_pits_L)
g_L_R.darrays[0].data = total_density
gio.write(g_L_R, OUTPUT+filename_L_R_gii)
"""
database = 'databaseBV'
cohort = 'hcp'
OUTPUT = '/media/yl247234/SAMSUNG/'+cohort+'/'+database+'/pits_density/'
path = '/media/yl247234/SAMSUNG/'+cohort+'/'+database+'/'+cohort+'/'
temp_file_s_ids='/home/yl247234/s_ids_lists/s_ids.json'
with open(temp_file_s_ids, 'r') as f:
    data = json.load(f)
s_ids  = list(json.loads(data))

#total_density = np.array([])
pits_data_R2 = np.array([])
pits_data_L2 = np.array([])
count_R2 = 0
count_L2 = 0
for s_id in s_ids:
    path_s =  path+s_id+'/'
    file_pits_R = path_s + 't1mri/BL/default_analysis/segmentation/mesh/surface_analysis_sym/'+s_id+'_Rwhite_pits_smoothed0.7_60_on_atlas.gii'
    file_pits_L = path_s + 't1mri/BL/default_analysis/segmentation/mesh/surface_analysis_sym/'+s_id+'_Lwhite_pits_smoothed0.7_60_on_atlas.gii'
    if os.path.isfile(file_pits_R):
        count_R2 +=1
        if pits_data_R2.size == 0:
            pits_data_R2 = gio.read(file_pits_R).darrays[0].data
        else:
            pits_data_R2 += gio.read(file_pits_R).darrays[0].data
    if os.path.isfile(file_pits_L):
        count_L2 +=1       
        if pits_data_L2.size == 0:
            pits_data_L2 = gio.read(file_pits_L).darrays[0].data
        else:
            pits_data_L2 += gio.read(file_pits_L).darrays[0].data


pits_data_R_temp = pits_data_R2/count_R2
pits_data_L_temp = pits_data_L2/count_L2

total_density = (pits_data_R_temp+pits_data_L_temp)/2
g_L_R = gio.read(file_pits_L)
g_L_R.darrays[0].data = total_density
gio.write(g_L_R, OUTPUT+filename_L_R_gii)


OUTPUT = '/media/yl247234/SAMSUNG/all_data/'+database+'/pits_density/'

pits_data_R_temp = (pits_data_R+pits_data_R2)/(count_R+count_R2)
pits_data_L_temp = (pits_data_L+pits_data_L2)/(count_L+count_L2)

total_density = (pits_data_R_temp+pits_data_L_temp)/2
g_L_R = gio.read(file_pits_L)
g_L_R.darrays[0].data = total_density
gio.write(g_L_R, OUTPUT+filename_L_R_gii)


"""
