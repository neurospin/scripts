"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re
import nibabel.gifti.giftiio as gio

OUTPUT = '/neurospin/imagen/workspace/cati/BVdatabase/'
filename_R = 'R_average_pits_smoothed.txt'
filename_L = 'L_average_pits_smoothed.txt'
filename_R_gii = 'R_average_pits_smoothed.gii'
filename_L_gii = 'L_average_pits_smoothed.gii'

path = '/neurospin/imagen/workspace/cati/BVdatabase/'

centres = ['Berlin','Dresden', 'Dublin', 'Hamburg', 'London', 'Mannheim', 'Nottingham', 'Paris']
"""subjects_R = np.loadtxt('/neurospin/imagen/workspace/cati/subject_redo_smooth_pits_R_prev.txt')
subjects_R= ['%012d' % int(i) for i in subjects_R]"""
pits_data_R = np.array([])
pits_data_L = np.array([])
count_R = 0
count_L = 0
for centre in centres:
    path_c = path+centre+'/'
    for subject in os.listdir(path_c):
        path_s =  path_c+subject+'/'
        if os.path.isdir(path_s):
            file_pits_R = path_s + 't1mri/BL/default_analysis/segmentation/mesh/surface_analysis/'+subject+'_Rwhite_pits_smoothed_on_atlas.gii'
            file_pits_L = path_s + 't1mri/BL/default_analysis/segmentation/mesh/surface_analysis/'+subject+'_Lwhite_pits_smoothed_on_atlas.gii'
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

pits_data_R = pits_data_R/count_R
pits_data_L = pits_data_L/count_L


g_R = gio.read(file_pits_R)
g_L = gio.read(file_pits_L)
g_R.darrays[0].data = pits_data_R
g_L.darrays[0].data = pits_data_L
np.savetxt(OUTPUT+filename_R, pits_data_R)
np.savetxt(OUTPUT+filename_L, pits_data_L)
gio.write(g_R, OUTPUT+filename_R_gii)
gio.write(g_L, OUTPUT+filename_L_gii)
