# -*- coding: utf-8 -*-

"""
Created on Tuesday May 13th 2014

@author: hl237680

Crée un tableau mettant en correspondance le nom du sujet (patients classés
dans le bon ordre) et les valeurs d'intensité des deux points chauds
correspondants dans les images originales.
On obtient ainsi un phénotype.
"""

import os
import csv
import numpy as np
import nibabel as ni
from glob import glob


PROJECT_DIR = '/neurospin/brainomics/2013_imagen_anat_vgwas_gpu'
ORIGIN_IMG_DIR = os.path.join(PROJECT_DIR, '2013_imagen_bmi', 'data',
                              'VBM', 'new_segment_spm8')
SORTED_SUBJECT_LIST = os.path.join(PROJECT_DIR, 'data',
                                   'sorted_subject_list.npy')
PHENOTYPE_FINAL = os.path.join(PROJECT_DIR, 'documents', '2014jan24_Plink',
                               'phenotype_final_rs13107325.csv')
INIMG_FILENAME_TEMPLATE = 'smwc1{subject_id:012}*.nii'


subject_all = np.load(SORTED_SUBJECT_LIST)

#phenotype = np.zeros((len(subject_all), 4))

origin_img_list = []
    
for s in subject_all:
    pattern = ORIGIN_IMG_DIR + '/' + INIMG_FILENAME_TEMPLATE
    pattern = pattern.format(subject_id = int(s))
    if len(glob(pattern)) == 1:
        origin_img_list.append(glob(pattern)[0])
    else:
        print "ERROR cannot find ", s

#Coordinates of the right hot spot (fslview)
z1 = 51
y1 = 75
x1 = 44

#Coordinates of the left hot spot (fslview)
z2 = 52
y2 = 76
x2 = 75


fp = open(PHENOTYPE_FINAL, 'wb')
cw = csv.writer(fp, delimiter=' ')
cw.writerow(['FID','IID','pR','pL'])
for i, s in enumerate(subject_all.tolist()):
    img = ni.load(origin_img_list[i])
    img_data = img.get_data()
    tmp = []
    #Family ID
    tmp.append("%012d" %(int(subject_all[i])))
    #Individual ID
    tmp.append("%012d" %(int(subject_all[i])))
    #Intensity value of the right hot spot
    tmp.append("%f" %(img.get_data()[x1,y1,z1]))
    #Intensity value of the left hot spot    
    tmp.append( "%f" %(img.get_data()[x2,y2,z2]))
#    print tmp
    cw.writerow(tmp)
fp.close()



#fp = open(PHENOTYPE_FINAL, 'wb')
#cw = csv.writer(fp, delimiter=' ')
#cw.writerow(['FID','IID','pR','pL'])
#for i, s in enumerate(subject_all.tolist()):
#    img = ni.load(origin_img_list[i])
#    img_data = img.get_data()
#    img_max = np.max(img_data)
#    [x2,y2,z2] = np.where(img_data == img_max)     #get max in an automized way
#    np.cast[np.int](x2,y2,z2)
#    x1 = 145 - x2   #par symétrie axiale
#    tmp = []
#    #Family ID
#    tmp.append("%012d" %(int(subject_all[i])))
#    #Individual ID
#    tmp.append("%012d" %(int(subject_all[i])))
#    #Intensity value of the right hot spot
#    tmp.append("%f" %(img_data()[x1,y2,z2]))
#    #Intensity value of the left hot spot    
#    tmp.append( "%f" %(img_data()[x2,y2,z2]))
##    print tmp
#    cw.writerow(tmp)
#fp.close()
