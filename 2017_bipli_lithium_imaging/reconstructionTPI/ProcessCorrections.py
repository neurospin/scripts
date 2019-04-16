#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 15:59:02 2018

@author: js247994
"""

import nibabel as nib
from nibabel.affines import apply_affine
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from lmfit import Model

def funcT2(x, a, b,c):
    return (a * (np.exp(-b * x)) + c)

def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")


def Fieldmap_to_Source_space(source_shape,source_affine,fieldmap_file):
    
    fieldmap_img = nib.load(fieldmap_file)
    fieldmap_img_data= fieldmap_img.get_data()
    fieldmap_shape=fieldmap_img_data.shape
    fieldmap_affine=fieldmap_img.affine
    
    fieldmap_to_source_vox=np.linalg.inv(source_affine).dot(fieldmap_affine)
    #anat_vox_center = (np.array(anat_img_data.shape) - 1) / 2
    fieldmap_interpoled=np.zeros(source_shape)
    fieldmap_weight=np.zeros(source_shape)
    for i in range(0,fieldmap_shape[0]):
        for j in range(0,fieldmap_shape[1]):
            for k in range(0,fieldmap_shape[2]):
                position_new=np.round(apply_affine( fieldmap_to_source_vox, ([i,j,k]) ))
                position_new=position_new.astype(int)
                position_new=(position_new[0],position_new[1],position_new[2])
                fieldmap_interpoled[position_new]=fieldmap_interpoled[position_new]+fieldmap_img_data[i,j,k]
                fieldmap_weight[position_new]=fieldmap_weight[position_new]+1
    fieldmap_interpoled=np.nan_to_num(fieldmap_interpoled/fieldmap_weight)
    return(fieldmap_interpoled)
    
    #--fieldmap /neurospin/ciclops/projects/BIPLi7/ClinicalData/Processed_Data/2017_02_21/Field_mapping_2/field_mapping_phase.nii
    
def Fieldmap_get(fieldmap_file):
    
    fieldmap_img = nib.load(fieldmap_file)
    fieldmap_img_data= fieldmap_img.get_data()
    return(fieldmap_img_data)
    
def T2starestimate(datamat,echoes):
    data_amp=np.sqrt(datamat.real**2+datamat.imag**2)
    datasorted=np.partition(data_amp.flatten(), -2)
    numechoes=np.size(echoes)
    Sigvals=np.zeros(numechoes)
    numtests=1
    for i in range(numtests):
        maxival=datasorted[-(i+1)];
        maxiind=np.where(data_amp==maxival)
        for numecho in range(numechoes):
            Sigvals[numecho]=data_amp[maxiind[0],maxiind[1],numecho,maxiind[3]]
            
        xdata=echoes
        ydata=Sigvals  
        y_test = funcT2(xdata, float(200), 1/62, 0)
        #np.random.seed(1729)
        #y_noise = 0.2 * np.random.normal(size=xdata.size)
        #ydata_test = y_test + y_noise
        plt.plot(xdata, ydata, 'b-', label='data')
        plt.plot(xdata,y_test,'r-', label='data')
        popt, pcov = curve_fit(funcT2, xdata, ydata)
            
        #maxind=data_amp.argmax();
        #maxindarray=np.unravel_index([maxind],data.shape);
    
    #maxcompreal = arr[arr.real.argmax()]  # complex value with maximum real part
#maxcomp = arr.max()  # complex value with maximum real part, same as above
#maxcompimag = arr[arr.imag.argmax()]
