# -*- coding:Utf-8 -*-
#
# Author : Arthur Coste
# Date : December 2014
# Purpose : Handle NIfTI format
#--------------------------------------------------------------------------------------------------------------------------------- 

#import os

def SaveArrayAsNIfTI(Matrix,affine,OutputFileName):
      
	print(('INFO    : Saving image to : ', OutputFileName))
	import nibabel as nib
	# print ResoX
	# print affine
	img = nib.Nifti1Image(Matrix, affine)
	# print img.get_header()
	nib.save(img,OutputFileName)
	print("------------------------------------------------------------")
	

	
def GetMatrixFromNIfTI(imageNIfTI):

	import nibabel as nib
	matrix=nib.load(imageNIfTI)
	return matrix
	
def SaveArrayAsNIfTI_2(Matrix,ResoX,ResoY,ResoZ,NbPoints,NbLines,NbSlices,rad,orientation,OutputFileName):
	print(('INFO    : Saving Image to : ', OutputFileName))

	#from nipy.core.api import Image
	from nibabel.nifti1 import Nifti1Header
	import nibabel as nib
	import numpy as np

	if rad:
		if (str(orientation)==str("Transverse")):
			print((Matrix.shape))
			Matrix=np.reshape(Matrix,(NbPoints,NbPoints,NbSlices))
			print((Matrix.shape))
			affine = np.diag([ResoX,ResoY,ResoZ,1])
		if (str(orientation)==str("Coronal")):
			print((Matrix.shape))
			Matrix=np.reshape(Matrix,(NbPoints,NbSlices,NbPoints))
			print((Matrix.shape))
			affine = np.diag([ResoX,ResoZ,ResoY,1])
		if (str(orientation)==str("Sagital")):
			print((Matrix.shape))
			Matrix=np.reshape(Matrix,(NbSlices,NbPoints,NbPoints))
			print((Matrix.shape))
			affine = np.diag([ResoZ,ResoX,ResoY,1])

	else:
		if (str(orientation)==str("Transverse")):
			print((Matrix.shape))
			Matrix=np.reshape(Matrix,(NbPoints,NbLines,NbSlices))
			print((Matrix.shape))
			affine = np.diag([ResoX,ResoY,ResoZ,1])
		if (str(orientation)==str("Coronal")):
			print((Matrix.shape))
			Matrix=np.reshape(Matrix,(NbPoints,NbSlices,NbLines))
			print((Matrix.shape))
			affine = np.diag([ResoX,ResoZ,ResoY,1])
		if (str(orientation)==str("Sagital")):
			print((Matrix.shape))
			Matrix=np.reshape(Matrix,(NbSlices,NbPoints,NbLines))
			print((Matrix.shape))
			affine = np.diag([ResoZ,ResoX,ResoY,1])
	
	img = nib.Nifti1Image(Matrix,affine)
	nib.save(img,OutputFileName)
	print('------------------------------------------------------------')
    
def SaveArrayAsNIfTI_3(Matrix,ResoX,ResoY,ResoZ,OutputFileName):

	print(('INFO    : Saving image to : ', OutputFileName))
	
	from nipy.core.api import Image
	from nibabel.nifti1 import Nifti1Header
	import nibabel as nib
	import numpy as np
	# print ResoX
	affine = np.diag([ResoX,ResoY, ResoZ,  1])  # Homogeneous affine Matrix 
	# print affine
	img = nib.Nifti1Image(Matrix, affine)
	# print img.get_header()
	nib.save(img,OutputFileName)
	print("------------------------------------------------------------")    
