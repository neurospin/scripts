# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:54:44 2018

@author: JS247994
"""
# import the necessary packages
from skimage.measure import compare_ssim as ssim
#import matplotlib.pyplot as plt
import numpy as np
#import cv2
import nibabel as nib
import os
import matplotlib.pyplot as plt

from scipy import stats

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err


#def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	#m = mse(imageA, imageB)
	#s = ssim(imageA, imageB)
 
	# setup the figure
	#fig = plt.figure(title)
	#plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	# show first image
	#ax = fig.add_subplot(1, 2, 1)
	#plt.imshow(imageA, cmap = plt.cm.gray)
	#plt.axis("off")
 
	# show the second image
	#ax = fig.add_subplot(1, 2, 2)
	#plt.imshow(imageB, cmap = plt.cm.gray)
	#plt.axis("off")
 
	# show the images
	#plt.show()
    
    
    
    
# load the images -- the original, the original + contrast,
# and the original + photoshop

#original = cv2.imread("C:/Users/js247994/Pictures/AbstractImages/imagesnorm.png")
#contrast = cv2.imread("C:/Users/js247994/Pictures/AbstractImages/Fig1.png")
#shopped = cv2.imread("C:/Users/js247994/Pictures/Article_postrot/fig_subj_10_post_calib_Li_MGE_0_1.75_rot.png")
 
# convert the images to grayscale
#original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)
#contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

#TPI_test='V:/projects/BIPLi7/Clinicaldata/Processed_Data/2018_10_26/TPI/Reconstruct_gridding/06-MNIspace/MNI_Patient18_21deg_MID417_B0cor_KBgrid_MODULE_Echo0_TE300_rhoSSFP_filt.nii'
#Trufi_test='V:/projects/BIPLi7/Clinicaldata/Processed_Data/2018_10_26/Trufi/05-MNIspace/MNI_trufi_10000_rhoSSFP_test_filt.nii'
#TPI_test='V:/projects/BIPLi7/Clinicaldata/Processed_Data/2018_06_01/TPI/Reconstruct_gridding/06-MNIspace/MNI_Patient18_21deg_MID417_B0cor_KBgrid_MODULE_Echo0_TE300_rhoSSFP_filt.nii'
#Trufi_test='V:/projects/BIPLi7/Clinicaldata/Processed_Data/2018_06_01/Trufi/05-MNIspace/MNI_trufi_10000_rhoSSFP_test_filt.nii'
    
TPI_dir='V:\projects\BIPLi7\Clinicaldata\Analysis\ProcessingDecember_2\Processing_December_masked\TPI_Lithiumfiles_01'
Trufi_dir='V:\projects\BIPLi7\Clinicaldata\Analysis\ProcessingDecember_2\Processing_December_masked\Trufi_Lithiumfiles_01'

#TPI_dir='V:\projects\BIPLi7\Clinicaldata\Analysis\ProcessingDecember_2\Processing_December_quantif\TPI_Lithiumfiles_01'
#Trufi_dir='V:\projects\BIPLi7\Clinicaldata\Analysis\ProcessingDecember_2\Processing_December_quantif\Trufi_Lithiumfiles_01'

#TPI_dir='V:\projects\BIPLi7\Clinicaldata\Analysis\ProcessingDecember\Processing_December_quantif\TPI_Lithiumfiles_01'
#Trufi_dir='V:\projects\BIPLi7\Clinicaldata\Analysis\ProcessingDecember\Processing_December_quantif\Trufi_Lithiumfiles_01'

TPI_dir='V:\projects\BIPLi7\Clinicaldata\Analysis\ProcessingDecember_2\Processing_December_uniform\TPI_Lithiumfiles_01'
Trufi_dir='V:\projects\BIPLi7\Clinicaldata\Analysis\ProcessingDecember_2\Processing_December_uniform\Trufi_Lithiumfiles_01'

TPI_dir='V:\projects\BIPLi7\Clinicaldata\Analysis\ProcessingDecember_2\Processing_December_uniform\Trufi_Lithiumfiles_01'
Trufi_dir='V:\projects\BIPLi7\Clinicaldata\Analysis\ProcessingDecember_2\Processing_December_quantif\Trufi_Lithiumfiles_01'

TPI_dir='V:\projects\BIPLi7\Clinicaldata\Analysis\ProcessingDecember_2\Processing_December_uniform\TPI_Lithiumfiles_01'
Trufi_dir='V:\projects\BIPLi7\Clinicaldata\Analysis\ProcessingDecember_2\Processing_December_quantif\TPI_Lithiumfiles_01'

TPI_files=os.listdir(TPI_dir)
TPI_files=sorted(TPI_files)
TPI_files=TPI_files[10:]

Trufi_files=os.listdir(Trufi_dir)
Trufi_files=sorted(Trufi_files)
Trufi_files=Trufi_files[10:]

Masks_dir='V:\projects\BIPLi7\Masks\Regionmaps'
Masks_files=os.listdir(Masks_dir)
Masks_files=sorted(Masks_files)
num_masks=np.size(Masks_files)
num_patients=np.size(Trufi_files)
num_patients=9
x=np.zeros(num_patients,)
y=np.zeros(num_patients,)
X=np.zeros(num_masks*num_patients,)
Y=np.zeros(num_masks*num_patients,)
err=np.zeros(num_masks*num_patients,)
n=[]

dic={}


        
    
    
i=0
mask_num=0
symbols=['+','o','*','.','x','s','d','v']
colors=['r','g','b','c','m','y','k','k']
colors=['gold','r','magenta','indigo','darkorange','green','blue','cyan']
fig, ax = plt.subplots()
for mask_file in Masks_files:
    maskpath=os.path.join(Masks_dir,mask_file)
    Mask_path_nib = nib.load(maskpath)
    Mask_path_data= Mask_path_nib.get_data()
    Mask_path_data= np.squeeze(Mask_path_data)
    Mask_path_data[np.isnan(Mask_path_data)] = 0
    Mask_path_shape=Mask_path_data.shape
    Mask_path_affine=Mask_path_nib.affine
    
    #ssim_vals[i]=ssim(TPI_masked,Trufi_masked)
    patient_num=0
    
    for TPI_file,Trufi_file in zip(TPI_files,Trufi_files):
        #print(TPI_file)
        #print(Trufi_file)
        TPI_file_path=os.path.join(TPI_dir,TPI_file)
        Trufi_file_path=os.path.join(Trufi_dir,Trufi_file)
        
        TPI_file_nib = nib.load(TPI_file_path)
        TPI_file_data= TPI_file_nib.get_data()
        TPI_file_shape=TPI_file_data.shape
        TPI_file_affine=TPI_file_nib.affine
        TPI_file_data[np.isnan(TPI_file_data)] = 0
        TPI_file_data[TPI_file_data<0]=0
        #TPI_file_data=TPI_file_data/np.mean(TPI_file_data)
    
        Trufi_file_nib = nib.load(Trufi_file_path)
        Trufi_file_data= Trufi_file_nib.get_data()
        Trufi_file_shape=Trufi_file_data.shape
        Trufi_file_affine=Trufi_file_nib.affine
        Trufi_file_data[np.isnan(Trufi_file_data)] = 0
        Trufi_file_data=Trufi_file_data
        Trufi_file_data[Trufi_file_data<0]=0
        
        Trufi_masked=Trufi_file_data[Mask_path_data>0]
        TPI_masked=TPI_file_data[Mask_path_data>0]
        #dic={(TPI_file,Trufi_file,mask_file):(np.mean(TPI_masked),np.mean(Trufi_masked))}
        #x[patient*num_masks+mask_num]=np.mean(TPI_masked)
        #y[patient*num_masks+mask_num]=np.mean(Trufi_masked)
        x[patient_num]=np.mean(TPI_masked)
        y[patient_num]=np.mean(Trufi_masked) 
        X[i]=np.mean(TPI_masked)
        Y[i]=np.mean(Trufi_masked)  
        err[i]=x[patient_num]-y[patient_num]
        n.append(TPI_file)
        patient_num=patient_num+1
        i=i+1
        #Trufi_file_data=Trufi_file_data/np.mean(Trufi_file_data)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    print(mask_file)
    print(slope, intercept, r_value, p_value, std_err)
    print('done')    
    plt.scatter(x,y,color=colors[mask_num],facecolors='none',marker='o',label=mask_file.split('.')[0])
    #xr=np.arange(0,0.6,0.1)
    #yr=xr*1+0
    #plt.plot(xr,yr,'r--')
    #plt.grid(True)
    #plt.show()
    #for i, txt in enumerate(n):
    #    ax.annotate(txt, (x[i], y[i]))


    mask_num=mask_num+1

slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
error_true=np.sqrt(np.sum(err**2))
print('overall calculation')
print(slope, intercept, r_value, p_value, std_err)
#plt.legend(loc='upper left')    
x=np.arange(0,0.6,0.1)
#y=x*slope+intercept
#intercept=0
#slope=0.83
y=x*slope+intercept  #for uniform quantif
#y=x*1.02699+0.02536
#y=x*0.78  # Trufi uniform vs trufi region
#y=x*0.62068+0.047186

plt.plot(x,y,'r--')
x=np.arange(0,0.6,0.1)
#y=x*slope+intercept
y=x*1+0   #for uniform quantif
#○plt.plot(x,y,'b:')
plt.grid(True)
plt.xlabel('[Li]* SSFP with uniform quantif (in mmol/L)')
plt.ylabel('[Li]* SSFP with regional quantif (in mmol/L)')
slopecut="%.2f" % slope
plottext=r'slope='+str(slopecut)
plt.text(0.10,0.42,plottext)
ax.set_xlim([0,0.6])
ax.set_ylim([0,0.6])
plt.show()