# -*- coding:Utf-8 -*-
# Author : Arthur Coste
# Date : December 2014
# Purpose : Regrid data  
#---------------------------------------------------------------------------------------------------------------------------------

import os
#import argparse
import math #,scipy.ndimage
import numpy as np
from scipy.interpolate import griddata
from visualization import PlotImg,PlotImgMag,PlotReconstructedImage,PlotImg2,PlotImgMag2,DefineROIonImage
#from DataFilter import *

#from ctypes import *
#STD_OUTPUT_HANDLE_ID = c_ulong(0xfffffff5)
#windll.Kernel32.GetStdHandle.restype = c_ulong
#std_output_hdl = windll.Kernel32.GetStdHandle(STD_OUTPUT_HANDLE_ID)

def ind2sub( sizes, index, num_indices ):
    """
    Map a scalar index of a flat 1D array to the equivalent
    d-dimensional index
    Example:
    | 1  4  7 |      | 1,1  1,2  1,3 |
    | 2  5  8 |  --> | 2,1  2,2  2,3 |
    | 3  6  9 |      | 3,1  3,2  3,3 |
    """

    denom = num_indices
    num_dims = sizes.shape[0]
    multi_index = np.empty( ( num_dims ), np.int32 )
    for i in range( num_dims - 1, -1, -1 ):
        denom /= sizes[i]
        multi_index[i] = index / denom
        index = index % denom
    return multi_index

def sub2ind( sizes, multi_index ):
    """
    Map a d-dimensional index to the scalar index of the equivalent flat array
    Example:
    | 1,1  1,2  1,3 |     | 1  4  7 | 
    | 2,1  2,2  2,3 | --> | 2  5  8 |
    | 3,1  3,2  3,3 |     | 3  6  9 |      
    """
    num_dims = sizes.shape[0]
    index = 0
    shift = 1
    for i in range( num_dims ):
        index += shift * multi_index[i]
        shift *= sizes[i]
    return index

def Direct_Reconstruction_2D(NbLines,NbPoints,NbAverages,NbCoils,NbSlice,OverSamplingFactor,Nucleus,MagneticField,SubSampling,Data):
	
	print ('------------------------------------------------------------')
	print ('INFO    : Running 2D Cartesian Reconstruction')
	print ()
	if Data.size == 0: 
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 12)
		print ('ERROR    : Empty Data Frame')
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
		
	if NbCoils==0 : NbCoils=1
	print (Data.shape)
	DataAvg=np.sum(Data[:,:,:,:,:],1) # sum over all averages
	
	if Data.size != 0 and NbLines and NbPoints and NbAverages and NbCoils and NbSlice and OverSamplingFactor:
		Average_Combined_Kspace = np.zeros(shape=(int(NbSlice),int(NbCoils),int(NbLines),int(NbPoints)*int(OverSamplingFactor)), dtype=np.complex64)
		Coil_Combined_Kspace = np.zeros(shape=(int(NbSlice),int(NbLines),int(NbPoints)*int(OverSamplingFactor)))
		PlotImgMag2(np.absolute(DataAvg[0][0][:][:]))
		# Begining of modifications for Jacques's data
		DataAvg[:,:,:,125:128]=0
		PlotImgMag2(np.absolute(DataAvg[0][0][:][:]))
		for i in range(NbSlice):
			DataAvg[i,0,0:16,:]=np.flipud(DataAvg[i,0,0:16,:])
		PlotImgMag2(np.absolute(DataAvg[3][0][:][:]))
		
		# temp=DataAvg[0,0,0:(NbLines/2),:];
		# DataAvg[0,0,0:(NbLines/2),:]=np.fliplr(np.conjugate(DataAvg[:,:,NbLines/2:NbLines,:]));
		# DataAvg[0,0,NbLines/2:NbLines,:]=np.fliplr(np.conjugate(np.flipud(temp)))
		# PlotImgMag2(np.absolute(DataAvg[0][0][:][:]))
		for j in range(NbSlice):
			for i in range(NbCoils):
				# Average_Combined_Kspace[j][i][:][:]=Average_Combined_Kspace[j][i][:][:]+ np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.squeeze(DataAvg[j][i][:][:]))))
				Average_Combined_Kspace[j][i][:][:]=Average_Combined_Kspace[j][i][:][:]+ np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.squeeze(DataAvg[j][i][:][:]))))
			Coil_Combined_Kspace[j][:][:]=Coil_Combined_Kspace[j][:][:]+np.absolute(Average_Combined_Kspace[j][i][:][:])*np.absolute(Average_Combined_Kspace[j][i][:][:])
			Coil_Combined_Kspace[j][:][:]=np.sqrt((Coil_Combined_Kspace[j][:][:]))
			# PlotReconstructedImage((Coil_Combined_Kspace[j,32:96,32:96]))
			# PlotReconstructedImage((Coil_Combined_Kspace[j,:,64:192]))
			PlotReconstructedImage((Coil_Combined_Kspace[j,:,48:80])) # for Jacques's data
			# PlotReconstructedImage((Coil_Combined_Kspace[j,:,:])) 
	print ('INFO    : Reconstruction Computed')
	print ('------------------------------------------------------------')
	
	# return Coil_Combined_Kspace[j,32:96,32:96]
	
	return Coil_Combined_Kspace[j,:,48:80]  # for Jacques's data
	# return Coil_Combined_Kspace[j,:,:]
	
def Direct_Reconstruction_3D(NbLines,NbPoints,NbAverages,NbCoils,NbSlice,OverSamplingFactor,Nucleus,MagneticField,SubSampling,Data):
	print ('------------------------------------------------------------')
	print ('INFO    : Running 3D Cartesian Reconstruction')
	print ()
	if Data.size == 0: 
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 12)
		print ('ERROR    : Empty Data Frame')
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
	print (Data.shape)
	DataAvg=np.sum(Data[:,:,:,:,:],1) # sum over all averages	
	print (DataAvg.shape)
	Coil_Combined_Kspace = np.zeros(shape=(int(NbSlice),int(NbLines),int(NbPoints)*int(OverSamplingFactor)))
	for i in range(NbSlice):
		for j in range(NbCoils):
				# for k in range(NbLines): 
					# Kspace[i][j][k][:] = Data[i][j][k][:]
			Coil_Combined_Kspace[:][:][:]+np.absolute(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(np.squeeze(DataAvg[i][j][:][:][:])))))**2
	print (Coil_Combined_Kspace.shape)
	PlotReconstructedImage((Coil_Combined_Kspace[4,:,:]))	
	# if Data.size != 0 and NbLines and NbPoints and NbAverages and NbCoils and NbSlice and OverSamplingFactor:
		# Average_Combined_Kspace = np.zeros(shape=(int(NbCoils),int(NbLines),int(NbPoints)*int(OverSamplingFactor)), dtype=np.complex64)
		# Coil_Combined_Kspace = np.zeros(shape=(int(NbLines),int(NbPoints)*int(OverSamplingFactor)))
		# for i in range(NbCoils):
			# for k in range(NbAverages):
				# Average_Combined_Kspace[i][:][:]=Average_Combined_Kspace[i][:][:]+ np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(np.squeeze(Data[k][i][:][:]))))
			# Coil_Combined_Kspace[:][:]=Coil_Combined_Kspace[:][:]+np.absolute(Average_Combined_Kspace[i][:][:])*np.absolute(Average_Combined_Kspace[i][:][:])
		# Coil_Combined_Kspace[:][:]=np.sqrt((Coil_Combined_Kspace[:][:]))
		# PlotImgMag2((Coil_Combined_Kspace))
	return Coil_Combined_Kspace
	
def Radial_Regridding(NbLines,NbPoints,NbAverages,NbCoils,NbSlice,OverSamplingFactor,Nucleus,MagneticField,SubSampling,Data,Interpolation,verbose,Gmax=None,BW=None):

	print ('------------------------------------------------------------')
	print ('INFO    : Running 2D RADIAL Regridding Version 1.2')
	print ()
		
	if BW==None:
		try:
			#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 15)
			BW=float(input('REQUEST : BandWidth per pixel = '))
			#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
		except ValueError:
			print ('Not a valid number')
	
	if Gmax==None:
		try:
			#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 15)
			Gmax=float(input('REQUEST : ReadOut Gradient Amplitude = '))
			#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
		except ValueError:
			print ('Not a valid number')
	
	if NbLines: 
		print ('INFO    : Number of Lines = ',NbLines)
	else: print ('ERROR    : Unspecified number of radial lines')
	if NbPoints: 
		print ('INFO    : Number of Points per line = ',NbPoints)
	else: print ('ERROR    : Unspecified number of points per line')
	if NbAverages: 
		print ('INFO    : Number of Averages = ',NbAverages)
	else: print ('ERROR    : Unspecified number of Averages')
	if NbCoils: 
		print ('INFO    : Number of Coils = ',NbCoils)
	else: print ('ERROR    : Unspecified number of Coils')
	if Nucleus:
		if Nucleus == "1H": 
			Gamma = 42.576e6
		if str(Nucleus) == '23Na': 
			Gamma=11.262e6
		if Nucleus == "31P": 
			Gamma= 17.235e6
		print ('INFO    : Used Nucleus = ',Nucleus)
		print ('INFO    : Gyromagnetic Ratio = ',Gamma,'Hz')
	else: print ('ERROR    : Unspecified Nucleus')
	if MagneticField: 
		print ('INFO    : Magnetic Field Strength : ',MagneticField,'T')
	else : print ('ERROR    : Undefined magnetic Field value')
	if SubSampling: 
		SubSampling = True
		print ('INFO    : SubSampling is Enabled')
	else: SubSampling=False; print  ('INFO    : SubSampling is Disabled')
	
	print ('INFO    : BandWidth  = ',BW, 'Hz / pix')
	print ('INFO    : ReadOut Gradient Amplitude  = ',Gmax, 'mT / m')
	Gmax=Gmax*1e-3
	RO_Amp=[NbLines]
	PE_Amp=[NbLines]
	rampdur=400e-6
	# BW=400.0
	Tobs=1/BW
	Tech=Tobs/(NbPoints*OverSamplingFactor)
	NbPts_On_Slope=round(rampdur/Tech)

	if Data.size != 0 and NbLines and NbPoints and NbAverages and NbCoils and NbSlice and OverSamplingFactor:
		print("INFO    : Generating K space locations",end="\t\t")
		for i in range(NbLines+1):   
			RO_Amp.append(Gmax*math.cos((i)*2*math.pi/(NbLines)))
			PE_Amp.append(Gmax*math.sin((i)*2*math.pi/(NbLines)))

		RO_GradShape = np.zeros(shape=(int(NbLines),int(NbPoints)*OverSamplingFactor))
		PE_GradShape = np.zeros(shape=(int(NbLines),int(NbPoints)*OverSamplingFactor))
		for i in range(NbLines):
			for j in range(NbPoints*OverSamplingFactor):
				# /!\ WARNING : I used a python vector for RO/PE_Amp... Maybe not that smart, but the first element is its size !
				if j< NbPts_On_Slope or j==NbPts_On_Slope:
					RO_GradShape[i][j]=Gamma*RO_Amp[i+1]/rampdur*(((j)*Tech)*((j)*Tech))/2
					PE_GradShape[i][j]=Gamma*PE_Amp[i+1]/rampdur*(((j)*Tech)*((j)*Tech))/2
				else:
					RO_GradShape[i][j]=Gamma*RO_Amp[i+1]*(((((NbPts_On_Slope)*(Tech))**2))/2/rampdur+(j-NbPts_On_Slope)*Tech)
					PE_GradShape[i][j]=Gamma*PE_Amp[i+1]*(((((NbPts_On_Slope)*(Tech))**2))/2/rampdur+(j-NbPts_On_Slope)*Tech)
		print("[DONE]")
		
		x = np.linspace(np.amin(RO_GradShape), np.amax(RO_GradShape),NbPoints*OverSamplingFactor*2)
		y = np.linspace(np.amin(PE_GradShape), np.amax(PE_GradShape),NbPoints*OverSamplingFactor*2)
		xv, yv = np.meshgrid(x, y)
		
		if verbose: print ('dims of ReadOut Gradient Matrix        = ',len(RO_GradShape),'x',len(RO_GradShape[0]))
		if verbose: print ('dims of Phase Encoding Gradient Matrix = ',len(PE_GradShape),'x',len(PE_GradShape[0]))
		if verbose: print ('dims of Output Grid X                  = ',len(xv),'x',len(xv[0]))
		if verbose: print ('dims of Output Grid Y                  = ',len(yv),'x',len(yv[0]))
		if verbose: print ('dims of Input values                   = ',len(Data[0][0][0][:]),'x',len(Data[0][0][0][0][:]))
		
		print("INFO    : Performing regridding",end="\t\t\t")
		
		Regridded_kspace = np.zeros(shape=(int(NbSlice),int(NbAverages),int(NbCoils),int(len(xv)),int(len(xv[0]))), dtype=np.complex64)		
		Average_Combined_Kspace = np.zeros(shape=(int(NbSlice),int(NbCoils),int(len(xv)),int(len(xv[0]))), dtype=np.complex64)
		Coil_Combined_Kspace = np.zeros(shape=(int(NbSlice),int(len(xv)),int(len(xv[0]))))
		
		RO_GradShape_V=np.zeros(RO_GradShape.size)
		RO_GradShape_V=np.reshape(RO_GradShape,RO_GradShape.size)
		PE_GradShape_V=np.zeros(PE_GradShape.size)
		PE_GradShape_V=np.reshape(PE_GradShape,PE_GradShape.size)
		xv_V=np.zeros(xv.size)
		xv_V=np.reshape(xv,xv.size)
		yv_V=np.zeros(yv.size)
		yv_V=np.reshape(yv,yv.size)
		
		for j in range(NbSlice):
			for i in range(int(NbCoils)):
				for k in range(NbAverages):
					Data_V=np.zeros(Data[j][k][i].size)
					Data_V=np.reshape(Data[j][k][i],Data[j][k][i].size)
					Regridded_kspace[j][k][i]=np.reshape(griddata((RO_GradShape_V,PE_GradShape_V),Data_V,(xv_V, yv_V),method=str(Interpolation),fill_value=0),(int(len(xv)),int(len(xv[0]))))
					Average_Combined_Kspace[j][i][:][:]=Average_Combined_Kspace[j][i][:][:]+ np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.squeeze(Regridded_kspace[j][k][i][:][:]))))
					# PlotReconstructedImage(np.absolute(Average_Combined_Kspace[j][i]))
				Coil_Combined_Kspace[j][:][:]=Coil_Combined_Kspace[j][:][:]+np.absolute(Average_Combined_Kspace[j][i][:][:])**2
			Coil_Combined_Kspace[j][:][:]=np.sqrt((Coil_Combined_Kspace[j][:][:]))
			PlotReconstructedImage((Coil_Combined_Kspace[j]))
		PlotReconstructedImage(np.absolute((Regridded_kspace[0][0][0])))
		print("[DONE]")
		
		if SubSampling:
			print ("Performing SubSampling")

		print ('------------------------------------------------------------')
	else:
		print ('ERROR MISSING ARGUMENT')
	
	# return Coil_Combined_Kspace,Average_Combined_Kspace[0,:,:,:]
	return Coil_Combined_Kspace

def TPI_Regridding(NbLines,NbPoints,NbAverages,NbCoils,NbSlice,OverSamplingFactor,Nucleus,MagneticField,SubSampling,Data,KX,KY,KZ,Interpolation,verbose):

	print ('------------------------------------------------------------')
	print ('INFO    : Running 3D TPI Regridding Version 1.0')
	print ()
	
	# ro       = mrprot.Meas.BaseResolution;										Contained in NbPoints x Oversampling Factor
	# nChannels = mrprot.Meas.iMaxNoOfRxChannels;									= Number of Coils
	# res      = mrprot.MeasYaps.sWiPMemBlock.adFree{1}; # [mm]						TO GET !!
	# p        = 0.01 * mrprot.MeasYaps.sWiPMemBlock.adFree{2}; # [%] 				TO GET !!
	# fov      = mrprot.MeasYaps.sSliceArray.asSlice{1,1}.dReadoutFOV; # [mm] 		Get from ACQParams
	# kmax =   1/(2*res); #1/mm														TO compute
	
	if Data.size != 0 and NbLines and NbPoints and NbAverages and NbCoils and NbSlice and OverSamplingFactor:
		print ('INFO    : Importing TPI Trajectory') 
		print (np.amin(KX), np.amax(KX))
		print (np.amin(KY), np.amax(KY))
		print (np.amin(KZ), np.amax(KZ))
		
		x = np.linspace(np.amin(KX), np.amax(KX),NbPoints*OverSamplingFactor/8)
		y = np.linspace(np.amin(KY), np.amax(KY),NbPoints*OverSamplingFactor/8)
		z = np.linspace(np.amin(KZ), np.amax(KZ),NbPoints*OverSamplingFactor/8)
		xv, yv, zv = np.meshgrid(x, y, z)
		
		Regridded_kspace = np.zeros(shape=(int(NbAverages),int(NbCoils),int(len(xv[0])),int(len(xv[0])),int(len(xv[0]))), dtype=np.complex64)		
		Average_Combined_Kspace = np.zeros(shape=(int(NbCoils),int(len(xv[0])),int(len(xv[0])),int(len(xv[0]))), dtype=np.complex64)
		Coil_Combined_Kspace = np.zeros(shape=(int(len(xv[0])),int(len(xv[0])),int(len(xv[0]))))
		
		X_GradShape_V=np.zeros(KX.size)
		X_GradShape_V=np.reshape(KX,KX.size)
		Y_GradShape_V=np.zeros(KY.size)
		Y_GradShape_V=np.reshape(KY,KY.size)
		Z_GradShape_V=np.zeros(KZ.size)
		Z_GradShape_V=np.reshape(KZ,KZ.size)
		xv_V=np.zeros(xv.size)
		xv_V=np.reshape(xv,xv.size)
		yv_V=np.zeros(yv.size)
		yv_V=np.reshape(yv,yv.size)
		zv_V=np.zeros(zv.size)
		zv_V=np.reshape(zv,zv.size)
		
		print (X_GradShape_V.shape, Y_GradShape_V.shape, Z_GradShape_V.shape)
		print (xv_V.shape, yv_V.shape, zv_V.shape)
		print (Data.shape)
		
		for i in range(NbCoils):
			for k in range(NbAverages):
				print ('starting regridding coil ',i)
				Regridded_kspace[k][i]=np.reshape(griddata((X_GradShape_V,Y_GradShape_V,Z_GradShape_V),Data,(xv_V, yv_V,zv_V),method=str(Interpolation),fill_value=0),(int(len(xv[0])),int(len(xv[0])),int(len(xv[0]))))
				# print('combining')
				Average_Combined_Kspace[i][:][:][:]=Average_Combined_Kspace[i][:][:][:]+ np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(np.squeeze(Regridded_kspace[k][i][:][:][:]))))
			Coil_Combined_Kspace[:][:][:]=Coil_Combined_Kspace[:][:][:]+np.absolute(Average_Combined_Kspace[i][:][:][:])*np.absolute(Average_Combined_Kspace[i][:][:][:])
		Coil_Combined_Kspace[:][:][:]=np.sqrt((Coil_Combined_Kspace[:][:][:]))

	else:
		print ('ERROR MISSING ARGUMENT')
	return Coil_Combined_Kspace
	
def CalculateKaiserBesselKernel(width, overgriddingfactor, length):	
	
	if not length:
		length=32
	if length <2:
		length = 2
	a=overgriddingfactor
	w=width
	
	beta=np.pi*np.sqrt((w**2/a**2)*((a-0.5)**2)-0.8)		#Rapid Gridding Reconstruction With a Minimal Oversampling Ratio P. J. Beatty et al, 2005
	u=np.arange(0,length-1,dtype=float)/(length-1) * (int(w)/2)
	kernel=beta*np.sqrt(1-((2*u[np.absolute(u) < int(w)/2])/w)**2) 	#Selection of a Convolution Function for Fourier Inversion using Griding, Jackson et al 1991
	KBkernel=(np.i0(kernel))/width
	# KBkernel=KBkernel/KBkernel[0]
	print (KBkernel)
	
	return KBkernel, u,beta
	
def simple_poly_area(x, y):
    # For short arrays (less than about 100 elements) it seems that the
    # Python sum is faster than the np sum. Likewise for the Python
    # built-in abs.
    return .5 * abs(sum(x[:-1] * y[1:] - x[1:] * y[:-1]) +
                    x[-1] * y[0] - x[0] * y[-1])
  	
	
def KaiserBesselRegridding(NbLines,NbPoints,NbAverages,NbCoils,NbSlice,OverSamplingFactor,Nucleus,MagneticField,SubSampling,Data,Interpolation,verbose,Gmax=None,BW=None):

	if NbCoils==0 : NbCoils=1
	print (NbCoils)
	#Compute Kaiser Bessel
	NormalizedKernel, u, beta = CalculateKaiserBesselKernel(3,2,4)
	NormalizedKernelflip=np.flipud(NormalizedKernel)
	# print((NormalizedKernelflip))
	NormalizedKernelflip=np.delete(NormalizedKernelflip,2)
	NormalizedKernel=np.append(NormalizedKernelflip,NormalizedKernel)
	print((NormalizedKernel))
	NormalizedKernel=np.mat(NormalizedKernel)
	NormalizedKernel2D=np.transpose(NormalizedKernel)*(NormalizedKernel)	
	print(NormalizedKernel2D)
	print(NormalizedKernel2D.shape)

	
	#Perform Gridding
	
	#1) Generate K space trajectory
	RO_Amp=[NbLines]
	PE_Amp=[NbLines]
	RO_Amp=[NbLines]
	PE_Amp=[NbLines]

	if BW==None:
		try:
			#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 15)
			BW=float(input('REQUEST : BandWidth per pixel = '))
			#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
		except ValueError:
			print ('Not a valid number')
	
	if Gmax==None:
		try:
			#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 15)
			Gmax=float(input('REQUEST : ReadOut Gradient Amplitude = '))
			#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
		except ValueError:
			print ('Not a valid number')
			
	if NbLines: 
		print ('INFO    : Number of Lines = ',NbLines)
	else: print ('ERROR    : Unspecified number of radial lines')
	if NbPoints: 
		print ('INFO    : Number of Points per line = ',NbPoints)
	else: print ('ERROR    : Unspecified number of points per line')
	if NbAverages: 
		print ('INFO    : Number of Averages = ',NbAverages)
	else: print ('ERROR    : Unspecified number of Averages')
	if NbCoils: 
		print ('INFO    : Number of Coils = ',NbCoils)
	else: print ('ERROR    : Unspecified number of Coils')
	if Nucleus:
		if Nucleus == "1H": 
			Gamma = 42.576e6
		if str(Nucleus) == '23Na': 
			Gamma=11.262e6
		if Nucleus == "31P": 
			Gamma= 17.235e6
		if Nucleus == "7Li":
			Gamma = 16.546e6
		print ('INFO    : Used Nucleus = ',Nucleus)
		print ('INFO    : Gyromagnetic Ratio = ',Gamma,'Hz')
	else: print ('ERROR    : Unspecified Nucleus')
	if MagneticField: 
		print ('INFO    : Magnetic Field Strength : ',MagneticField,'T')
	else : print ('ERROR    : Undefined magnetic Field value')
	if SubSampling: 
		SubSampling = True
		print ('INFO    : SubSampling is Enabled')
	else: SubSampling=False; print  ('INFO    : SubSampling is Disabled')
	
	print ('INFO    : BandWidth  = ',BW, 'Hz / pix')
	print ('INFO    : ReadOut Gradient Amplitude  = ',Gmax, 'mT / m')
	Gmax=Gmax*1e-3
	rampdur=400e-6
	Tobs=1/BW
	Tech=Tobs/(NbPoints*OverSamplingFactor)
	NbPts_On_Slope=round(rampdur/Tech)		
	
	if (str(Nucleus)==str('31P')): Gamma=17.235e6
	if (str(Nucleus)==str('1H')): Gamma = 42.576e6
	print("INFO    : Generating K space locations",end="\t\t")
	for i in range(NbLines+1):   
		RO_Amp.append(Gmax*math.cos((i)*2*math.pi/(NbLines)))
		PE_Amp.append(Gmax*math.sin((i)*2*math.pi/(NbLines)))

	RO_GradShape = np.zeros(shape=(int(NbLines),int(NbPoints)*OverSamplingFactor))
	PE_GradShape = np.zeros(shape=(int(NbLines),int(NbPoints)*OverSamplingFactor))
	a=np.zeros(int(NbPoints)*OverSamplingFactor*int(NbLines))
	b=np.zeros(int(NbPoints)*OverSamplingFactor*int(NbLines))
	for i in range(NbLines):
		for j in range(NbPoints*OverSamplingFactor):
			# /!\ WARNING : I used a python vector for RO/PE_Amp... Maybe not that smart, but the first element is its size !
			if j< NbPts_On_Slope or j==NbPts_On_Slope:
				RO_GradShape[i][j]=Gamma*RO_Amp[i+1]/rampdur*(((j)*Tech)*((j)*Tech))/2
				PE_GradShape[i][j]=Gamma*PE_Amp[i+1]/rampdur*(((j)*Tech)*((j)*Tech))/2
				# a=np.append(a,RO_GradShape[i][j])
				# b=np.append(b,PE_GradShape[i][j])
				
			else:
				RO_GradShape[i][j]=Gamma*RO_Amp[i+1]*(((((NbPts_On_Slope)*(Tech))**2))/2/rampdur+(j-NbPts_On_Slope)*Tech)
				PE_GradShape[i][j]=Gamma*PE_Amp[i+1]*(((((NbPts_On_Slope)*(Tech))**2))/2/rampdur+(j-NbPts_On_Slope)*Tech)
				# a=np.append(a,RO_GradShape[i][j])
				# b=np.append(b,PE_GradShape[i][j])
	print("[DONE]")
	
	from scipy.spatial import Voronoi, voronoi_plot_2d
	import matplotlib.pyplot as plt

	# points = np.column_stack((a,b))
	points = np.column_stack((np.ravel(RO_GradShape),np.ravel(PE_GradShape)))
	del(a);del(b)
	# compute Voronoi tesselation
	vor = Voronoi(points)
	# voronoi_plot_2d(vor)
	# plt.show()
	# print (len(vor.regions))
	area = np.zeros(len(vor.regions))
	for node in range(len(vor.regions)):
		xv = vor.vertices[vor.regions[node],0]
		yv = vor.vertices[vor.regions[node],1]
		# print (xv,yv)
		if ((xv !=[] and yv !=[]) and (len(xv)==4)):
			area[node] = simple_poly_area(xv, yv)
			# print ("Aire ",node," == ",simple_poly_area(xv, yv))
	weightVoronoi=np.unique(np.round(area,9))
	weightVoronoi=np.sort(weightVoronoi)
	weightVoronoi=weightVoronoi+(1/(int(NbPoints)*OverSamplingFactor))
	weightVoronoi=weightVoronoi/max(weightVoronoi)
	weightVoronoi=np.append(weightVoronoi,1)
	weightVoronoi[0]=1e-5
	del(points);del(vor);del(area);del(xv);del(yv);

	# print (weightVoronoi)
	print (len(weightVoronoi))
	if not os.path.isfile("voronoi_coefs.txt") :
		np.savetxt("voronoi_coefs.txt",weightVoronoi)
	
	LinearWeigths=np.linspace(1/(NbPoints*OverSamplingFactor), 1,NbPoints*OverSamplingFactor)
	LinearWeigths=np.delete(LinearWeigths,0)
	LinearWeigths=np.append(LinearWeigths,1)
	# print (LinearWeigths)
	# print (len(LinearWeigths))
	if not os.path.isfile("linear_coefs.txt") :
		np.savetxt("linear_coefs.txt",LinearWeigths)

	# size = np.round((np.amax(PE_GradShape)-np.amin(PE_GradShape))/2)
	# size = NbPoints*OverSamplingFactor*2
	imsize = NbPoints*OverSamplingFactor*2
	# size = np.round(NbPoints*OverSamplingFactor*2*2)
	size = np.round(NbPoints*OverSamplingFactor*2*2)
	# On utilise une grille 2 fois plus grande pour minimiser les erreurs de regridding et on crop apr\E8s
	# size = NbPoints*OverSamplingFactor*2*2
	print(size,np.amax(PE_GradShape),np.amin(PE_GradShape))
	
	# Regridded_kspace = np.zeros(shape=(int(NbSlice),int(NbAverages),int(NbCoils),int(size),int(size)), dtype=np.complex64)		
	# Average_Combined_Kspace = np.zeros(shape=(int(NbSlice),int(NbCoils),int(size),int(size)), dtype=np.complex64)
	# Coil_Combined_Kspace = np.zeros(shape=(int(NbSlice),int(size),int(size)))
	
	# x_current=np.round((RO_GradShape/np.amax(RO_GradShape))*size/2)
	# y_current=np.round((PE_GradShape/np.amax(PE_GradShape))*size/2)
	print (Data.shape)
	DataAvg = np.zeros(shape=(int(NbSlice),int(NbCoils),int(size),int(size)), dtype=np.complex64)
	
	# IF CODE CRASHES THEN IT MEANS THAT DATA HAVE ALREADY BEEN SUMMED IN HIGHER FUNCTIONS
	DataAvg=np.sum(Data[:,:,:,:,:],1) # sum over all averages
	# DataAvg=np.sum(Data[:,1:2:,:,:],1)
	# DataAvg=np.mean(Data,1)
	print (DataAvg.shape)

	# Regridded_kspace = np.zeros(shape=(int(NbSlice),int(NbCoils),int(size)+2,int(size)+2), dtype=np.complex64)		
	# Coil_Combined_Kspace = np.zeros(shape=(int(NbSlice),int(size)+2,int(size)+2))
	Regridded_kspace = np.zeros(shape=(int(NbSlice),int(NbCoils),int(size),int(size)), dtype=np.complex64)		
	Coil_Combined_Kspace = np.zeros(shape=(int(NbSlice),int(size),int(size)), dtype=np.complex64)
	Coil_Combined_Kspace_Module = np.zeros(shape=(int(NbSlice),int(size),int(size)))
	Coil_Combined_Kspace_Phase = np.zeros(shape=(int(NbSlice),int(size),int(size)))
	usedline=0
	KspaceNRJ=0.0
	
	# N=20
	# chosen_golden_lines=np.zeros(N)
	# goldenAngle = 180/1.618;
	# for g in range(N):
		# chosen_golden_lines[g] = np.unique(np.mod(round(g*goldenAngle),360)) ;
	# print (chosen_golden_lines)
	
	for j in range (NbSlice):
		for i in range(int(NbCoils)):
			for l in range(NbLines):
				# We generate a random value (Uniform Distribution (Gaussian ?)) and compare it with some threshold to remove the line
				rand= np.random.rand(1)
				
				# current_angle = l*360/402;
				# if sum(round(current_angle)==chosen_golden_lines)==1:
				if rand[0] >= 0.0 : 
				# if np.mod(l,2)==0:
					usedline+=1
					rand2= np.random.rand(1)
					if rand2 >=0 :
						nbofpoints=NbPoints*OverSamplingFactor
					else :
						nbofpoints=NbPoints
					for m in range(nbofpoints):
						# Pour avoir le K space centr\E9
						# x_current=np.round(RO_GradShape[l][m]/np.amax(RO_GradShape)*size/2)-size/2+1
						# y_current=np.round(PE_GradShape[l][m]/np.amax(PE_GradShape)*size/2)-size/2+1
						x_current=np.round(RO_GradShape[l][m]/np.amax(RO_GradShape)*size/2)
						y_current=np.round(PE_GradShape[l][m]/np.amax(PE_GradShape)*size/2)
						# Val=DataAvg[j][i][l][m]
						Val=DataAvg[j][i][l][m]*weightVoronoi[m]
						# Val=Data[j][i][l][m]*LinearWeigths[m]
						# print(Val)
						KspaceNRJ+=(float(np.absolute(Val)**2)*(1/float(BW)))
						
						for a in range(-1,1,1):
							for b in range(-1,1,1):
								# Regridded_kspace[j][i][y_current[l,m]+a][x_current[l,m]+b]=Regridded_kspace[j][i][y_current[l,m]+a][x_current[l,m]+b]+Val*NormalizedKernel2D[a+1,b+1]
								Regridded_kspace[j][i][y_current+a][x_current+b]=Regridded_kspace[j][i][y_current+a][x_current+b]+Val*NormalizedKernel2D[a+1,b+1]*np.exp(-i*(np.pi/4)*1j)
								# Regridded_kspace[j][i][-(y_current+a)][-(x_current+b)]=np.conjugate(Regridded_kspace[j][i][y_current+a][x_current+b])
								# Regridded_kspace[j][i][y_current+a][x_current+b]=Regridded_kspace[j][i][y_current+a][x_current+b]+Val*NormalizedKernel2D[a+1,b+1]
						# Regridded_kspace[j][i][y_current][x_current]=Regridded_kspace[j][i][y_current][x_current]+Val
						# Regridded_kspace[j][i][y_current[l,m]][x_current[l,m]]=Regridded_kspace[j][i][y_current[l,m]][x_current[l,m]]+Val
			Coil_Combined_Kspace[j]=np.sum(Regridded_kspace,1)
			Coil_Combined_Kspace_Module[j][:][:]=Coil_Combined_Kspace_Module[j][:][:]+np.absolute(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.squeeze(Regridded_kspace[j][i][:][:])))))**2
			# Coil_Combined_Kspace_Phase[j][:][:]=Coil_Combined_Kspace_Phase[j][:][:]+np.angle(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.squeeze(Regridded_kspace[j][i][:][:])))))
		Coil_Combined_Kspace_Module[j][:][:]=np.sqrt((Coil_Combined_Kspace_Module[j][:][:]))
		
		Coil_Combined_Kspace_Phase[j]=np.angle(np.fft.fftshift(np.fft.ifft2((np.squeeze(Coil_Combined_Kspace[j])))))
		PlotReconstructedImage(Coil_Combined_Kspace_Phase[j])
		# C = InverseFunc(len(Coil_Combined_Kspace_Module[0]),beta,3)
		# C = InverseFunc(len(Coil_Combined_Kspace_Module[0])/2,beta,3)
		# PlotReconstructedImage(Coil_Combined_Kspace_Module[j,imsize/2:imsize+imsize/2,imsize/2:imsize+imsize/2])	
		PlotReconstructedImage(Coil_Combined_Kspace_Module[j,np.round((size-imsize)/2):imsize+np.round((size-imsize)/2),np.round((size-imsize)/2):imsize+np.round((size-imsize)/2)])	
		PlotReconstructedImage(Coil_Combined_Kspace_Module[j])	
		# PlotReconstructedImage((Coil_Combined_Kspace_Module[j,imsize/2:imsize+imsize/2,imsize/2:imsize+imsize/2])/C)
	
	PlotImgMag(np.absolute((Regridded_kspace[0][0])))
	print("Reconstructed with ", float((float(usedline)/float(NbLines*NbCoils))*100), " % of acquired lines")
	from processingFunctions import ComputeFullImageSNR
	ImageSNR = ComputeFullImageSNR(Coil_Combined_Kspace_Module[j])
	print ("Estimated SNR of Image = ", ImageSNR, "dB")
	print ("K space NRJ = ", KspaceNRJ)
	print ("checksum = ",sum(sum(sum(Coil_Combined_Kspace_Module))))
	print("[DONE]")
	
	# return Coil_Combined_Kspace_Module[j][size / 4: -size / 4, size / 4: - size / 4]
	# return Coil_Combined_Kspace_Module[j,imsize/2:imsize+imsize/2,imsize/2:imsize+imsize/2]
	return Coil_Combined_Kspace_Module[j,np.round((size-imsize)/2):imsize+np.round((size-imsize)/2),np.round((size-imsize)/2):imsize+np.round((size-imsize)/2)],Coil_Combined_Kspace_Phase[j,np.round((size-imsize)/2):imsize+np.round((size-imsize)/2),np.round((size-imsize)/2):imsize+np.round((size-imsize)/2)]
	# return Coil_Combined_Kspace_Module,Coil_Combined_Kspace_Phase[j,np.round((size-imsize)/2):imsize+np.round((size-imsize)/2),np.round((size-imsize)/2):imsize+np.round((size-imsize)/2)]
	
def InverseFunc(size,beta,width):
	C = np.zeros(shape=(size,size))
	A= np.zeros(size)

	print(beta)
	print (size)
				
	for i in range (-int(round(size/2)),int(round(size/2)),1):
			if i !=0:
				A[i]=(np.sin(np.sqrt((np.pi**2)*(width**2)*((i)**2)-beta**2))/(np.sqrt((np.pi**2)*(width**2)*((i)**2)-beta**2)))
			if i ==0: 
				A[i]=np.mean([A[int(round(size/2))-2],A[int(round(size/2))-1],A[int(round(size/2))+1],A[int(round(size/2))+2]])	
	
	# for i in range (int(size)):
		# if (i-size/2) !=0:
			# A[i-size/2]=(np.sin(np.sqrt((np.pi**2)*(width**2)*((i-size/2)**2)-(beta)**2)))/(np.sqrt((np.pi**2)*(width**2)*((i-size/2)**2)-(beta)**2))
		# if (i-size/2) ==0: 
			# A[i]=1e-6
	# for i in range (-int(round(size/2)),int(round(size/2)),1):
		# if i !=0:
			# A[i-int(round(size/2))]=(np.sin(np.sqrt((np.pi**2)*(width**2)*((i)**2)-beta**2))/(np.sqrt((np.pi**2)*(width**2)*((i)**2)-beta**2)))
	# A[int(round(size/2))]=np.mean([A[-2+int(round(size/2))],A[-1+int(round(size/2))],A[1+int(round(size/2))],A[2+int(round(size/2))]])		
	A[0]=1
	A=np.absolute(A)
	# Af=np.flipud(A)
	# Af=np.delete(Af,int(size/2)-1)
	# A=np.append(Af,A)
	# print(A)
	# A=A/np.max(A)
	A=np.mat(A)
	C=np.transpose(A)*(A)		
	# print (C)
	print (C.shape)
	# PlotImgMag(C)
	
	return C	
	
def RegridBruker(NbLines,NbPoints,NbAverages,NbCoils,NbSlice,Nucleus,Data,Kx,Ky,Kz,coefX,coefY,coefZ,EffectivePoints,verbose):

	print ('------------------------------------------------------------')
	print ('INFO    : Regridding Bruker Data')
	print ()
	#Compute Kaiser Bessel
	# NormalizedKernel, u, beta = CalculateKaiserBesselKernel(3,2,4)
	# NormalizedKernelflip=np.flipud(NormalizedKernel)
	# print((NormalizedKernelflip))
	# NormalizedKernelflip=np.delete(NormalizedKernelflip,2)
	# NormalizedKernel=np.append(NormalizedKernelflip,NormalizedKernel)
	# print((NormalizedKernel))
	# NormalizedKernel=np.mat(NormalizedKernel)
	# NormalizedKernel2D=np.transpose(NormalizedKernel)*(NormalizedKernel)	
	# print(NormalizedKernel2D)
	# print(NormalizedKernel2D.shape)

	size = NbPoints*2

	Regridded_kspace = np.zeros(shape=(int(NbCoils),int(size),int(size),int(size)), dtype=np.complex64)		
	Coil_Combined_Kspace = np.zeros(shape=(int(size),int(size),int(size)))

	print (np.amin(Kx)*(size/1),np.amax(Kx)*(size/1))
	print (np.amin(Ky)*(size/1),np.amax(Ky)*(size/1))
	print (np.amin(Kz)*(size/1),np.amax(Kz)*(size/1))
	print (len(Kx))
	print (len(Ky))
	print (len(Kz))
	
	Kxloc=np.zeros(Kx.size);Kyloc=np.zeros(Ky.size);Kzloc=np.zeros(Kz.size);
	Kxloc=np.round(((Kx/np.amax(Kx))*((size-2)/2))-size/2)
	Kyloc=np.round(((Ky/np.amax(Ky))*((size-2)/2))-size/2)
	Kzloc=np.round(((Kz/np.amax(Kz))*((size-2)/2))-size/2)
	usedline=0
	for i in range(int(NbCoils)):
		for l in range(NbLines):
			if l%1000 ==0 : print (l)
			rand= np.random.rand(1)
			if rand[0] > 0.0 : 
				usedline+=1
				rand2= np.random.rand(1)
				if rand2 >0.0 :
					nbofpoints=NbPoints
				else :
					nbofpoints=NbPoints/2
				for m in range(nbofpoints):
					# x_current=int((np.round(Kx[(l*NbPoints)+m]/np.amax(Kx))*size/2)-size/2)
					# y_current=int((np.round(Ky[(l*NbPoints)+m]/np.amax(Ky))*size/2)-size/2)
					# z_current=int((np.round(Kz[(l*NbPoints)+m]/np.amax(Kz))*size/2)-size/2)
					# Val=Data[l][m]
					Val=Data[l][m]
					# for c in range(-1,1,1):
						# for a in range(-1,1,1):
							# for b in range(-1,1,1):
								# Regridded_kspace[k][i][z_current+c][y_current+a][x_current+b]=Regridded_kspace[k][i][z_current+c][y_current+a][x_current+b]+Val*NormalizedKernel2D[a+2,b+2]
					# Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]=Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]+Val
					Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]=Val
					# Regridded_kspace[i][-Kzloc[(l*NbPoints)+m]][-Kyloc[(l*NbPoints)+m]][-Kxloc[(l*NbPoints)+m]]=np.conj(Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]])
		# PlotImgMag(np.absolute((Regridded_kspace[0][0][48])))
		# Average_Combined_Kspace[i][:][:][:]=Average_Combined_Kspace[i][:][:][:]+ np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(np.squeeze(Regridded_kspace[k][i][:][:][:]))))
		Coil_Combined_Kspace[:][:][:]=Coil_Combined_Kspace[:][:][:]+np.absolute(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(np.squeeze(Regridded_kspace[i][:][:][:])))))**2
	Coil_Combined_Kspace[:][:][:]=np.sqrt((Coil_Combined_Kspace[:][:][:]))
	
	PlotReconstructedImage(Coil_Combined_Kspace[:][48][:])
	print (Regridded_kspace[0].shape)
	print("Reconstructed with ", float((float(usedline)/float(NbLines*NbCoils))*100), " % of acquired lines")
	
	return Coil_Combined_Kspace,np.absolute(Regridded_kspace[0]),np.angle(Regridded_kspace[0])
	
def KaiserBesselTPI(NbProjections,NbPoints,NbAverages,NbCoils,OverSamplingFactor,Nucleus,MagneticField,SubSampling,Data,KX,KY,KZ,pval,resolution,FOV,verbose,PSF_ND,PSF_D,B1sensitivity):
	NormalizedKernel, u, beta = CalculateKaiserBesselKernel(3,2,4)
	NormalizedKernelflip=np.flipud(NormalizedKernel)
	# print((NormalizedKernelflip))
	NormalizedKernelflip=np.delete(NormalizedKernelflip,2)
	NormalizedKernel=np.append(NormalizedKernelflip,NormalizedKernel)
	print((NormalizedKernel))

	if NbProjections: 
		print ('INFO    : Number of Lines = ',NbProjections)
	else: print ('ERROR    : Unspecified number of radial lines')
	if NbPoints: 
		print ('INFO    : Number of Points per line = ',NbPoints)
	else: print ('ERROR    : Unspecified number of points per line')
	if NbAverages: 
		print ('INFO    : Number of Averages = ',NbAverages)
	else: print ('ERROR    : Unspecified number of Averages')
	if NbCoils: 
		print ('INFO    : Number of Coils = ',NbCoils)
		if Nucleus != "1H" and Nucleus != "7Li":
			coilstart=1
		else :
			coilstart=0
	else: 
		print ('ERROR    : Unspecified number of Coils')
		coilstart=0
	if Nucleus:
		if Nucleus == "1H": 
			Gamma = 42.576e6
		if str(Nucleus) == '23Na': 
			Gamma=11.262e6
		if Nucleus == "31P": 
			Gamma= 17.235e6
		if Nucleus == "7Li":
			Gamma = 16.546e6
		print ('INFO    : Used Nucleus = ',Nucleus)
		print ('INFO    : Gyromagnetic Ratio = ',Gamma,'Hz')
	else: print ('ERROR    : Unspecified Nucleus')
	if MagneticField: 
		print ('INFO    : Magnetic Field Strength : ',MagneticField,'T')
	else : print ('ERROR    : Undefined magnetic Field value')
	
	print("INFO    : Reading K space locations")
	if KX.any() : 
		print("INFO    : KX [OK]")
		print('INFO    : KX bounds : ',np.amin(KX), np.amax(KX))
	else : print ("ERROR    :  [KX]") 
	if KY.any() : 
		print("INFO    : KY [OK]")
		print('INFO    : KY bounds : ',np.amin(KY), np.amax(KY))
	else : print ("ERROR    :  [KY]") 
	if KZ.any() : 
		print("INFO    : KZ [OK]")
		print('INFO    : KZ bounds : ',np.amin(KZ), np.amax(KZ))
	else : print ("ERROR    :  [KZ]") 
	
	
	if PSF_D: 
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 15)
		decay=float(input('REQUEST :  T2/T2* decay [us] >> '))
		ReadOutTime=float(input('REQUEST :  ReadOutTime [us] >> '))
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
		decroissance = np.zeros(NbPoints)
		for i in range(NbPoints):
			decroissance[i]=np.exp(-(i*ReadOutTime/NbPoints)/decay)
	
	NbCoils=int(NbCoils+1)
	# BW=260.0
	# size = NbPoints*OverSamplingFactor*2 # In this case we have twice the number of points (each line covers half of the plan)
	# On utilise une grille 2 fois plus grande pour minimiser les erreurs de regridding et on crop apr\E8s
	# size = NbPoints*OverSamplingFactor*2*2
	print (NbProjections)
	print(size)
	print (Data.shape)
	# DataAvg = np.zeros(shape=(int(NbCoils),int(NbProjections),int(size)), dtype=np.complex64)
	# DataAvg=np.sum(Data,0)
	# DataAvg=np.mean(Data,0)
	# print (DataAvg.shape)
	# size=size/16
	# size=np.round(size/4)
	# size=64
	size=round (FOV/resolution)
	#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 10)
	print ('INFO    : Reconstruction Matrix Size = ',size)
	#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
	
	# Kx_size=int(size/4)+4
	# Ky_size=int(size)+4
	# Kz_size=int(size*2)+4
	Kx_size=int(size)
	Ky_size=int(size)
	Kz_size=int(size)
	
	linweights=np.ones(NbPoints,dtype=float)
	print (linweights.shape)
	# for i in range(int(np.round(len(linweights)*float(pval)))):
	for i in range(len(linweights)/2):
		# linweights[i]=float(i)*float(1/(np.round(float(len(linweights))*float(pval))))+1e-04
		linweights[i]=np.power(float(i)*float(1/(np.round(float(len(linweights))*float(pval)))),1.25)+1e-04
		# linweights[i]=float(i**2)*float(1/96.0)+1e-04
		# print (linweights[i])
		
	Regridded_kspace = np.zeros(shape=(int(NbCoils),Kx_size,Ky_size,Kz_size), dtype=np.complex64)		
	Coil_Combined_Kspace = np.zeros(shape=(Kx_size,Ky_size,Kz_size))
	# Regridded_kspace = np.zeros(shape=(int(NbCoils),int(size),int(size),int(size)), dtype=np.complex64)		
	# Coil_Combined_Kspace = np.zeros(shape=(int(size),int(size),int(size)))
	usedline=0
	if (B1sensitivity): 
		sousech=0.95
	else : sousech=0.0
	# KspaceNRJ=0.0
	
	KX = np.ravel(KX); KY=np.ravel(KY); KZ=np.ravel(KZ)
	Kxloc=np.zeros(KX.size);Kyloc=np.zeros(KX.size);Kzloc=np.zeros(KX.size);
	# Kxloc=np.round(((KX/np.amax(KX))*((size)))-(size)-2)
	# Kyloc=np.round(((KY/np.amax(KY))*((size)/2))-(size/2)-2)
	# Kzloc=np.round(((KZ/np.amax(KZ))*((size)/8))-(size/8)-2)
	Kxloc=np.floor(((KX/np.amax(KX))*((size/2))-(size/2)+1))
	Kyloc=np.floor(((KY/np.amax(KY))*((size/2))-(size/2)+1))
	Kzloc=np.floor(((KZ/np.amax(KZ))*((size/2))-(size/2)+1))
	
	DensityCompensationCoefficients = np.zeros(NbPoints,dtype=float)
	for i in range (len(DensityCompensationCoefficients)):
		DensityCompensationCoefficients[i]=np.sqrt((KX[i+1]-KX[i])**2 + (KY[i+1]-KY[i])**2 + (KZ[i+1]-KZ[i])**2)*np.sqrt((KX[i]/np.amax(KX))**2 + (KY[i]/np.amax(KY))**2 + (KZ[i]/np.amax(KZ))**2)**2
		# print("2  :",np.sqrt((KX[i+1]-KX[i])**2 + (KY[i+1]-KY[i])**2 + (KZ[i+1]-KZ[i])**2))
	

	# if not os.path.isfile("Density_compensation.csv") :
		# f=open("Density_compensation.csv","w")
		# for i in range(len(DensityCompensationCoefficients)):
			# f.write(str(float(DensityCompensationCoefficients[i])))
			# f.write("\n")
		# f.close()
	# return		
	
	DensityCompensationCoefficients[int(NbPoints-1)]=DensityCompensationCoefficients[int(NbPoints-2)]
	DensityCompensationCoefficients[0]=DensityCompensationCoefficients[3]
	DensityCompensationCoefficients[1]=DensityCompensationCoefficients[3]
	DensityCompensationCoefficients[2]=DensityCompensationCoefficients[3]
	# print (DensityCompensationCoefficients)
	# return
	
	# Data=np.sum(Data[:,:,:,:],0)
	print (Data.shape)
	
	for i in range(coilstart,int(NbCoils)):
		for l in range(NbProjections):
			# We generate a random value (Uniform Distribution (Gaussian ?)) and compare it with some threshold to remove the line
			rand= np.random.rand(1)
			
			if rand[0] > sousech : 
				usedline+=1
				# print (rand[0])
				rand2= np.random.rand(1)
				if rand2 >0 :
					nbofpoints=NbPoints*OverSamplingFactor
				else :
					nbofpoints=NbPoints
				for m in range(nbofpoints):
					# Pour avoir le K space centr\E9
					# x_current=np.round(KX[l][m]/np.amax(KX)*size/2)-size/2-1
					# y_current=np.round(KY[l][m]/np.amax(KY)*size/2)-size/2-1
					# z_current=np.round(KZ[l][m]/np.amax(KZ)*size/2)-size/2-1
					# Val=DataAvg[i][l][m]
					if (PSF_ND and not PSF_D) : Val=1
					if (PSF_D and not PSF_ND)  : Val=decroissance[m]
					elif (not PSF_ND and not PSF_D) : Val=Data[i][l][m]*DensityCompensationCoefficients[m]
					# print (Val)
					# Val=DataAvg[j][i][l][m]*weightVoronoi[m]
					# Val=Data[j][i][l][m]*LinearWeigths[m]
					# print(Val)
					# KspaceNRJ+=(float(np.absolute(Val)**2)*(1/float(BW)))
					
					for a in range(-1,1,1):
						for b in range(-1,1,1):
							for c in range(-1,1,1):
								Regridded_kspace[i][Kzloc[(l*NbPoints)+m+a]][Kyloc[(l*NbPoints)+m+b]][Kxloc[(l*NbPoints)+m+c]]=Regridded_kspace[i][Kzloc[(l*NbPoints)+m+a]][Kyloc[(l*NbPoints)+m+b]][Kxloc[(l*NbPoints)+m+c]]+Val*NormalizedKernel[c+1]
					# Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]=Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]+Val
		Coil_Combined_Kspace[:][:][:]=Coil_Combined_Kspace[:][:][:]+np.absolute(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(np.squeeze(Regridded_kspace[i][:][:])))))**2
		print ('regridding coil', i+1)
		print ('used lines = ', usedline)
	Coil_Combined_Kspace[:][:][:]=np.sqrt((Coil_Combined_Kspace[:][:][:]))
	
	# PlotImgMag(np.absolute((Regridded_kspace[4][:][:])))
	# return Regridded_kspace[0]
	return Coil_Combined_Kspace
	
def KaiserBesselTPI_ME(NbProjections,NbPoints,NbAverages,NbCoils,OverSamplingFactor,Nucleus,MagneticField,SubSampling,Data,KX,KY,KZ,pval,resolution,FOV,verbose,PSF_ND,PSF_D,B1sensitivity,echoes,SaveKspace):
	NormalizedKernel, u, beta = CalculateKaiserBesselKernel(3,2,4)
	NormalizedKernelflip=np.flipud(NormalizedKernel)
	# print((NormalizedKernelflip))
	NormalizedKernelflip=np.delete(NormalizedKernelflip,2)
	NormalizedKernel=np.append(NormalizedKernelflip,NormalizedKernel)
	NormalizedKernel = NormalizedKernel/np.amax(NormalizedKernel)
	print((NormalizedKernel))

	if NbProjections: 
		print ('INFO    : Number of Lines = ',NbProjections)
	else: print ('ERROR    : Unspecified number of radial lines')
	if NbPoints: 
		print ('INFO    : Number of Points per line = ',NbPoints)
	else: print ('ERROR    : Unspecified number of points per line')
	if NbAverages: 
		print ('INFO    : Number of Averages = ',NbAverages)
	else: print ('ERROR    : Unspecified number of Averages')
	if NbCoils: 
		print ('INFO    : Number of Coils = ',NbCoils)
		if Nucleus != "1H":
			coilstart=1
		else :
			coilstart=0
	else: 
		print ('ERROR    : Unspecified number of Coils')
		coilstart=0
	if Nucleus:
		if Nucleus.find("1H")>-1:
			Gamma = 42.576e6
		if Nucleus.find("23Na")>-1:
			Gamma=11.262e6
		if Nucleus.find("31P")>-1:
			Gamma= 17.235e6
		if Nucleus.find("7Li")>-1:
			Gamma = 16.546e6
		print ('INFO    : Used Nucleus = ',Nucleus)
		print ('INFO    : Gyromagnetic Ratio = ',Gamma,'Hz')
	else: print ('ERROR    : Unspecified Nucleus')
	if MagneticField: 
		print ('INFO    : Magnetic Field Strength : ',MagneticField,'T')
	else : print ('ERROR    : Undefined magnetic Field value')
	
	print("INFO    : Reading K space locations")
	if KX.any() : 
		print("INFO    : KX [OK]")
		print('INFO    : KX bounds : ',np.amin(KX), np.amax(KX))
	else : print ("ERROR    :  [KX]") 
	if KY.any() : 
		print("INFO    : KY [OK]")
		print('INFO    : KY bounds : ',np.amin(KY), np.amax(KY))
	else : print ("ERROR    :  [KY]") 
	if KZ.any() : 
		print("INFO    : KZ [OK]")
		print('INFO    : KZ bounds : ',np.amin(KZ), np.amax(KZ))
	else : print ("ERROR    :  [KZ]") 
	
	
	if PSF_D: 
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 15)
		decay=float(input('REQUEST :  T2/T2* decay [us] >> '))
		ReadOutTime=float(input('REQUEST :  ReadOutTime [us] >> '))
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
		decroissance = np.zeros(NbPoints)
		for i in range(NbPoints):
			decroissance[i]=np.exp(-(i*ReadOutTime/NbPoints)/decay)
	
	NbCoils=int(NbCoils+1)
	# BW=260.0
	# size = NbPoints*OverSamplingFactor*2 # In this case we have twice the number of points (each line covers half of the plan)
	# On utilise une grille 2 fois plus grande pour minimiser les erreurs de regridding et on crop apr\E8s
	# size = NbPoints*OverSamplingFactor*2*2
	# print (NbProjections)
	# print(size)
	print (Data.shape)

	size=round (FOV/resolution)
	#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 10)
	print ('INFO    : Reconstruction Matrix Size = ',size)
	#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
	
	Kx_size=int(size)
	Ky_size=int(size)
	Kz_size=int(size)
	
	linweights=np.ones(NbPoints,dtype=float)
	# print (linweights.shape)
	# for i in range(int(np.round(len(linweights)*float(pval)))):
	for i in range(int(len(linweights)/2)):
		# linweights[i]=float(i)*float(1/(np.round(float(len(linweights))*float(pval))))+1e-04
		linweights[i]=np.power(float(i)*float(1/(np.round(float(len(linweights))*float(pval)))),1.25)+1e-04
		# linweights[i]=float(i**2)*float(1/96.0)+1e-04
		# print (linweights[i])
		
	Regridded_kspace = np.zeros(shape=(int(NbCoils),Kx_size,Ky_size,Kz_size), dtype=np.complex64)		
	Sum_of_Regridded_kspace = np.zeros(shape=(int(NbCoils),Kx_size,Ky_size,Kz_size), dtype=np.complex64)		
	Coil_Combined_Kspace_Module = np.zeros(shape=(echoes,Kx_size,Ky_size,Kz_size))
	Coil_Combined_Kspace_Phase = np.zeros(shape=(echoes,Kx_size,Ky_size,Kz_size))
	Abs_Sum_of_Regridded_kspace = np.zeros(shape=(Kx_size,Ky_size,Kz_size))
	# Regridded_kspace = np.zeros(shape=(int(NbCoils),int(size),int(size),int(size)), dtype=np.complex64)		
	# Coil_Combined_Kspace = np.zeros(shape=(int(size),int(size),int(size)))
	
	if (B1sensitivity): 
		sousech=0.95
	else : sousech=0.0
	# KspaceNRJ=0.0
	
	KX = np.ravel(KX); KY=np.ravel(KY); KZ=np.ravel(KZ)
	Kxloc=np.zeros(KX.size);Kyloc=np.zeros(KX.size);Kzloc=np.zeros(KX.size);
	# Kxloc=np.round(((KX/np.amax(KX))*((size)))-(size)-2)
	# Kyloc=np.round(((KY/np.amax(KY))*((size)/2))-(size/2)-2)
	# Kzloc=np.round(((KZ/np.amax(KZ))*((size)/8))-(size/8)-2)
	Kxloc=np.floor(((KX/np.amax(KX))*((size/2))-(size/2)+1))
	Kyloc=np.floor(((KY/np.amax(KY))*((size/2))-(size/2)+1))
	Kzloc=np.floor(((KZ/np.amax(KZ))*((size/2))-(size/2)+1))
	Kxloc=Kxloc.astype(int)   
	Kyloc=Kyloc.astype(int)   
	Kzloc=Kzloc.astype(int)   
	
	DensityCompensationCoefficients = np.zeros(NbPoints,dtype=float)
	for i in range (len(DensityCompensationCoefficients)):
		DensityCompensationCoefficients[i]=np.sqrt((KX[i+1]-KX[i])**2 + (KY[i+1]-KY[i])**2 + (KZ[i+1]-KZ[i])**2)*np.sqrt((KX[i]/np.amax(KX))**2 + (KY[i]/np.amax(KY))**2 + (KZ[i]/np.amax(KZ))**2)**2		
	
	DensityCompensationCoefficients[int(NbPoints-1)]=DensityCompensationCoefficients[int(NbPoints-2)]
	DensityCompensationCoefficients[0]=DensityCompensationCoefficients[3]
	DensityCompensationCoefficients[1]=DensityCompensationCoefficients[3]
	DensityCompensationCoefficients[2]=DensityCompensationCoefficients[3]
	
	# DensityCompensationCoefficients = np.zeros(shape=(NbProjections,NbPoints),dtype=float)
	# for p in range ((NbProjections)):
		# for i in range (NbPoints):
			# DensityCompensationCoefficients[p,i]=np.sqrt((KX[p*NbPoints+i+1]-KX[p*NbPoints+i])**2 + (KY[p*NbPoints+i+1]-KY[p*NbPoints+i])**2 + (KZ[p*NbPoints+i+1]-KZ[p*NbPoints+i])**2)*np.sqrt((KX[p*NbPoints+i]/np.amax(KX))**2 + (KY[p*NbPoints+i]/np.amax(KY))**2 + (KZ[p*NbPoints+i]/np.amax(KZ))**2)**2		
	
		# DensityCompensationCoefficients[p,int(NbPoints-1)]=DensityCompensationCoefficients[p,int(NbPoints-2)]
		# DensityCompensationCoefficients[p,0]=DensityCompensationCoefficients[p,3]
		# DensityCompensationCoefficients[p,1]=DensityCompensationCoefficients[p,3]
		# DensityCompensationCoefficients[p,2]=DensityCompensationCoefficients[p,3]
	
	# print (DensityCompensationCoefficients.shape)
	
	# Data=np.sum(Data[:,:,:,:],0)
	# print (Data.shape)
	if (len(Data.shape)) >4:
		Data=np.sum(Data[:,:,:,:,:],0)
	print (Data.shape)
	
	for echo in range (echoes):
		
		print ('>> Griding echo ',echo)
		for i in range(coilstart,int(NbCoils)):
			print ('   >> regridding coil', i+1)
			usedline=0
			for l in range(NbProjections):
				# We generate a random value (Uniform Distribution (Gaussian ?)) and compare it with some threshold to remove the line
				rand= np.random.rand(1)
				
				if rand[0] > sousech : 
					usedline+=1
					# print (rand[0])
					rand2= np.random.rand(1)
					if rand2 >0 :
						nbofpoints=NbPoints*OverSamplingFactor
					else :
						nbofpoints=NbPoints
					for m in range(nbofpoints):
						# Pour avoir le K space centr\E9
						# x_current=np.round(KX[l][m]/np.amax(KX)*size/2)-size/2-1
						# y_current=np.round(KY[l][m]/np.amax(KY)*size/2)-size/2-1
						# z_current=np.round(KZ[l][m]/np.amax(KZ)*size/2)-size/2-1
						# Val=DataAvg[i][l][m]
						if (PSF_ND and not PSF_D) : Val=1
						if (PSF_D and not PSF_ND)  : Val=decroissance[m]
						elif (not PSF_ND and not PSF_D) : Val=Data[i][l][echo][m]*DensityCompensationCoefficients[m]
						# print (np.squeeze(Data[i][l][echo][m]))
						# print (Val)
						# Val=DataAvg[j][i][l][m]*weightVoronoi[m]
						# Val=Data[j][i][l][m]*LinearWeigths[m]
						# print(Val)
						# KspaceNRJ+=(float(np.absolute(Val)**2)*(1/float(BW)))
						
						for a in range(-1,1,1):
							for b in range(-1,1,1):
								for c in range(-1,1,1):
									# print (Regridded_kspace[i][Kzloc[(l*NbPoints)+m+a]][Kyloc[(l*NbPoints)+m+b]][Kxloc[(l*NbPoints)+m+c]])
									# print (Val)
									Regridded_kspace[i][Kzloc[(l*NbPoints)+m+a]][Kyloc[(l*NbPoints)+m+b]][Kxloc[(l*NbPoints)+m+c]]=Regridded_kspace[i][Kzloc[(l*NbPoints)+m+a]][Kyloc[(l*NbPoints)+m+b]][Kxloc[(l*NbPoints)+m+c]]+Val*NormalizedKernel[c+1]
						# Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]=Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]+Val
			Sum_of_Regridded_kspace[:,:,:] = Sum_of_Regridded_kspace[:,:,:] + Regridded_kspace[:,:,:]		
			Coil_Combined_Kspace_Module[echo,:,:,:]=Coil_Combined_Kspace_Module[echo,:,:,:]+np.absolute(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(np.squeeze(Regridded_kspace[i][:][:])))))**2
			Coil_Combined_Kspace_Phase[echo,:,:,:]=Coil_Combined_Kspace_Phase[echo,:,:,:]+np.angle(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(np.squeeze(Regridded_kspace[i][:][:])))))**2
			Regridded_kspace = np.zeros(shape=(int(NbCoils),Kx_size,Ky_size,Kz_size), dtype=np.complex64)		#Jacques edit test
            
			print ('   >> Projections = ', float(int(usedline)/int(NbProjections))*100, '%')
		Coil_Combined_Kspace_Module[echo,:,:,:]=np.sqrt((Coil_Combined_Kspace_Module[echo,:,:,:]))
		#### Image Reorientation to Scanner DICOM anatomical Images
		## First we swap axes to bring the axial plane correctly opened in Anatomist compared to Anatomy
		## Second we rotate of 180\B0 to fix the left/Right and Antero Posterior Orientation vs Anatomy
		Coil_Combined_Kspace_Module[echo,:,:,:]=np.swapaxes(Coil_Combined_Kspace_Module[echo,:,:,:],0,2)
		Coil_Combined_Kspace_Module[echo,:,:,:]=np.swapaxes(Coil_Combined_Kspace_Module[echo,:,:,:],0,1)
		Coil_Combined_Kspace_Module[echo,:,:,:]=np.rot90(Coil_Combined_Kspace_Module[echo,:,:,:],2)
	print ('[done]')
	# PlotImgMag(np.absolute((Regridded_kspace[4][:][:])))
	# return Regridded_kspace[0]
	Abs_Sum_of_Regridded_kspace=np.absolute(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(np.squeeze(Sum_of_Regridded_kspace[:,:,:])))))**2
	if SaveKspace : return Coil_Combined_Kspace_Module, Coil_Combined_Kspace_Phase, Regridded_kspace
	else : return Coil_Combined_Kspace_Module, Coil_Combined_Kspace_Phase, Abs_Sum_of_Regridded_kspace


def KaiserBesselTPI_ME_B0(NbProjections,NbPoints,NbAverages,NbCoils,OverSamplingFactor,Nucleus,MagneticField,SubSampling,Data,KX,KY,KZ,pval,resolution,FOV,verbose,PSF_ND,PSF_D,B1sensitivity,echoes,TEs,SaveKspace,field_map,Timesampling,L,recon_method,ba_gridding):
	NormalizedKernel, u, beta = CalculateKaiserBesselKernel(3,2,4)
	NormalizedKernelflip=np.flipud(NormalizedKernel)
	# print((NormalizedKernelflip))
	NormalizedKernelflip=np.delete(NormalizedKernelflip,2)
	NormalizedKernel=np.append(NormalizedKernelflip,NormalizedKernel)
	NormalizedKernel = NormalizedKernel/np.amax(NormalizedKernel)
    
	print((NormalizedKernel))

	if NbProjections: 
		print ('INFO    : Number of Lines = ',NbProjections)
	else: print ('ERROR    : Unspecified number of radial lines')
	if NbPoints: 
		print ('INFO    : Number of Points per line = ',NbPoints)
	else: print ('ERROR    : Unspecified number of points per line')
	if NbAverages: 
		print ('INFO    : Number of Averages = ',NbAverages)
	else: print ('ERROR    : Unspecified number of Averages')
	if NbCoils: 
		print ('INFO    : Number of Coils = ',NbCoils)
		if Nucleus != "1H":
			coilstart=1
		else :
			coilstart=0
	else: 
		print ('ERROR    : Unspecified number of Coils')
		coilstart=0
	if Nucleus:
		GammaH = 42.576e6
		if Nucleus.find("1H")>-1:
			Gamma = 42.576e6      
		elif Nucleus.find("23Na")>-1:
			Gamma=11.262e6
		elif Nucleus.find("31P")>-1:
			Gamma= 17.235e6      
		elif Nucleus.find("7Li")>-1:
			Gamma = 16.546e6
		else: print ('ERROR    : Unrecognized Nucleus')            
		print ('INFO    : Used Nucleus = ',Nucleus)
		print ('INFO    : Gyromagnetic Ratio = ',Gamma,'Hz')
	else: print ('ERROR    : Unspecified Nucleus')
	if MagneticField: 
		print ('INFO    : Magnetic Field Strength : ',MagneticField,'T')
	else : print ('ERROR    : Undefined magnetic Field value')
	
	print("INFO    : Reading K space locations")
	if KX.any() : 
		print("INFO    : KX [OK]")
		print('INFO    : KX bounds : ',np.amin(KX), np.amax(KX))
	else : print ("ERROR    :  [KX]") 
	if KY.any() : 
		print("INFO    : KY [OK]")
		print('INFO    : KY bounds : ',np.amin(KY), np.amax(KY))
	else : print ("ERROR    :  [KY]") 
	if KZ.any() : 
		print("INFO    : KZ [OK]")
		print('INFO    : KZ bounds : ',np.amin(KZ), np.amax(KZ))
	else : print ("ERROR    :  [KZ]") 	
	
	if PSF_D: 
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 15)
		decay=float(input('REQUEST :  T2/T2* decay [us] >> '))
		ReadOutTime=float(input('REQUEST :  ReadOutTime [us] >> '))
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
		decroissance = np.zeros(NbPoints)
		for i in range(NbPoints):
			decroissance[i]=np.exp(-(i*ReadOutTime/NbPoints)/decay)
	
	NbCoils=int(NbCoils+1)
	# BW=260.0
	# size = NbPoints*OverSamplingFactor*2 # In this case we have twice the number of points (each line covers half of the plan)
	# On utilise une grille 2 fois plus grande pour minimiser les erreurs de regridding et on crop apr\E8s
	# size = NbPoints*OverSamplingFactor*2*2
	# print (NbProjections)
	# print(size)
	print (Data.shape)
	size=round (FOV/resolution)
	#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 10)
	print ('INFO    : Reconstruction Matrix Size = ',size)
	#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
	
	Kx_size=int(size)
	Ky_size=int(size)
	Kz_size=int(size)
	
	linweights=np.ones(NbPoints,dtype=float)
	# print (linweights.shape)
	# for i in range(int(np.round(len(linweights)*float(pval)))):
	for i in range(int(len(linweights)/2)):
		# linweights[i]=float(i)*float(1/(np.round(float(len(linweights))*float(pval))))+1e-04
		linweights[i]=np.power(float(i)*float(1/(np.round(float(len(linweights))*float(pval)))),1.25)+1e-04
		# linweights[i]=float(i**2)*float(1/96.0)+1e-04
		# print (linweights[i])      
	Demod_Data = np.zeros(shape=(np.shape(Data)+(L,)), dtype=np.complex64)
	Regridded_kspace = np.zeros(shape=(int(NbCoils),Kx_size,Ky_size,Kz_size,L), dtype=np.complex64)		
	Sum_of_Regridded_kspace = np.zeros(shape=(int(NbCoils),Kx_size,Ky_size,Kz_size,L), dtype=np.complex64)		
	Coil_Combined_Kspace_Module = np.zeros(shape=(echoes,Kx_size,Ky_size,Kz_size,L))
	Coil_Combined_temp = np.zeros(shape=(echoes,Kx_size,Ky_size,Kz_size))
	Coil_Combined_Kspace_Phase = np.zeros(shape=(echoes,Kx_size,Ky_size,Kz_size,L))
	Abs_Sum_of_Regridded_kspace = np.zeros(shape=(Kx_size,Ky_size,Kz_size,L))
	# Regridded_kspace = np.zeros(shape=(int(NbCoils),int(size),int(size),int(size)), dtype=np.complex64)		
	# Coil_Combined_Kspace = np.zeros(shape=(int(size),int(size),int(size)))
	
	if (B1sensitivity): 
		sousech=0.95
	else : sousech=0.0
	# KspaceNRJ=0.0

	#delta=(np.max(field_map) - np.min(field_map))/(L*2*np.pi) 
	deltaw = np.linspace(np.max(field_map), np.min(field_map), L)*2*np.pi
    
	time= np.linspace(0,Timesampling,NbPoints) 

  
	#dw_o_freq = deltaw[np.int(L/2)]    
	num_of_interp=1000
	tk=64
	timesamples=np.linspace(0,Timesampling,tk)
	#interpolation_omegas= np.linspace(np.min(field_map), np.max(field_map), num_of_interp)*2*np.pi
	#deltawi_tk=np.exp(1j * np.conj(timesamples * deltaw) )
	#y= np.exp(1j* timesamples.getH() * interpolation_omegas)

	KX = np.ravel(KX); KY=np.ravel(KY); KZ=np.ravel(KZ)
	Kxloc=np.zeros(KX.size);Kyloc=np.zeros(KX.size);Kzloc=np.zeros(KX.size);
	# Kxloc=np.round(((KX/np.amax(KX))*((size)))-(size)-2)
	# Kyloc=np.round(((KY/np.amax(KY))*((size)/2))-(size/2)-2)
	# Kzloc=np.round(((KZ/np.amax(KZ))*((size)/8))-(size/8)-2)
	Kxloc=np.floor(((KX/np.amax(KX))*((size/2))-(size/2)+1))
	Kyloc=np.floor(((KY/np.amax(KY))*((size/2))-(size/2)+1))
	Kzloc=np.floor(((KZ/np.amax(KZ))*((size/2))-(size/2)+1))
	Kxloc=Kxloc.astype(int)   
	Kyloc=Kyloc.astype(int)   
	Kzloc=Kzloc.astype(int)   
	
	DensityCompensationCoefficients = np.zeros(NbPoints,dtype=float)
	for i in range (len(DensityCompensationCoefficients)):
		DensityCompensationCoefficients[i]=np.sqrt((KX[i+1]-KX[i])**2 + (KY[i+1]-KY[i])**2 + (KZ[i+1]-KZ[i])**2)*np.sqrt((KX[i]/np.amax(KX))**2 + (KY[i]/np.amax(KY))**2 + (KZ[i]/np.amax(KZ))**2)**2		
		DensityCompensationCoefficients[i]=(np.sqrt((KX[i+1]-KX[i])**2 + (KY[i+1]-KY[i])**2 + (KZ[i+1]-KZ[i])**2)*np.sqrt((KX[i]/np.amax(KX))**2 + (KY[i]/np.amax(KY))**2 + (KZ[i]/np.amax(KZ))**2))**2		
	
	DensityCompensationCoefficients[int(NbPoints-1)]=DensityCompensationCoefficients[int(NbPoints-2)]
	DensityCompensationCoefficients[0]=DensityCompensationCoefficients[3]
	DensityCompensationCoefficients[1]=DensityCompensationCoefficients[3]
	DensityCompensationCoefficients[2]=DensityCompensationCoefficients[3]    
	# DensityCompensationCoefficients = np.zeros(shape=(NbProjections,NbPoints),dtype=float)
	# for p in range ((NbProjections)):
		# for i in range (NbPoints):
			# DensityCompensationCoefficients[p,i]=np.sqrt((KX[p*NbPoints+i+1]-KX[p*NbPoints+i])**2 + (KY[p*NbPoints+i+1]-KY[p*NbPoints+i])**2 + (KZ[p*NbPoints+i+1]-KZ[p*NbPoints+i])**2)*np.sqrt((KX[p*NbPoints+i]/np.amax(KX))**2 + (KY[p*NbPoints+i]/np.amax(KY))**2 + (KZ[p*NbPoints+i]/np.amax(KZ))**2)**2		
	
		# DensityCompensationCoefficients[p,int(NbPoints-1)]=DensityCompensationCoefficients[p,int(NbPoints-2)]
		# DensityCompensationCoefficients[p,0]=DensityCompensationCoefficients[p,3]
		# DensityCompensationCoefficients[p,1]=DensityCompensationCoefficients[p,3]
		# DensityCompensationCoefficients[p,2]=DensityCompensationCoefficients[p,3]	
	# print (DensityCompensationCoefficients.shape)
	# Data=np.sum(Data[:,:,:,:],0)
	# print (Data.shape)
	if (len(Data.shape)) >4:
		Data=np.sum(Data[:,:,:,:,:],0)
	print (Data.shape)
	
	for freq in range(L): #for freq in range(L):
		print(freq)
		for echo in range (echoes):
        		
        		print ('>> Griding echo ',echo)
        		for i in range(coilstart,int(NbCoils)):
        			print ('   >> regridding coil', i+1)
        			usedline=0
        			for l in range(NbProjections):
        				# We generate a random value (Uniform Distribution (Gaussian ?)) and compare it with some threshold to remove the line
        				rand= np.random.rand(1)
        				
        				if rand[0] > sousech : 
        					usedline+=1
        					# print (rand[0])
        					rand2= np.random.rand(1)
        					if rand2 >0 :
        						nbofpoints=NbPoints*OverSamplingFactor
        					else :
        						nbofpoints=NbPoints
        					for m in range(nbofpoints):
        						# Pour avoir le K space centr\E9
        						# x_current=np.round(KX[l][m]/np.amax(KX)*size/2)-size/2-1
        						# y_current=np.round(KY[l][m]/np.amax(KY)*size/2)-size/2-1
        						# z_current=np.round(KZ[l][m]/np.amax(KZ)*size/2)-size/2-1
        						# Val=DataAvg[i][l][m]
        						Demod_Data[i][l][echo][m][freq]= Data[i][l][echo][m]* np.exp(-1j * deltaw[freq] * (float(TEs[echo])/10**6+time[m]) )
        						if (PSF_ND and not PSF_D) : Val=1
        						if (PSF_D and not PSF_ND)  : Val=decroissance[m]
        						elif (not PSF_ND and not PSF_D) : Val=Demod_Data[i][l][echo][m][freq]*DensityCompensationCoefficients[m]
        						# print (np.squeeze(Data[i][l][echo][m]))
        						# print (Val)
        						# Val=DataAvg[j][i][l][m]*weightVoronoi[m]
        						# Val=Data[j][i][l][m]*LinearWeigths[m]
        						# print(Val)
        						# KspaceNRJ+=(float(np.absolute(Val)**2)*(1/float(BW)))
        						
        						for a in range(-1,1,1):
        							for b in range(-1,1,1):
        								for c in range(-1,1,1):
        									# print (Regridded_kspace[i][Kzloc[(l*NbPoints)+m+a]][Kyloc[(l*NbPoints)+m+b]][Kxloc[(l*NbPoints)+m+c]])
        									# print (Val)
        									try:                                            
        										Regridded_kspace[i,Kzloc[(l*NbPoints)+m+a],Kyloc[(l*NbPoints)+m+b],Kxloc[(l*NbPoints)+m+c],freq]=Regridded_kspace[i,Kzloc[(l*NbPoints)+m+a],Kyloc[(l*NbPoints)+m+b],Kxloc[(l*NbPoints)+m+c],freq]+Val*NormalizedKernel[c+1]
        									except:
        										print('hi')
        						# Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]=Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]+Val
        			Sum_of_Regridded_kspace[:,:,:,freq] = Sum_of_Regridded_kspace[:,:,:,freq] + Regridded_kspace[:,:,:,freq]		
        			Coil_Combined_Kspace_Module[echo,:,:,:,freq]=Coil_Combined_Kspace_Module[echo,:,:,:,freq]+np.absolute(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(np.squeeze(Regridded_kspace[i,:,:,:,freq])))))**2
        			Coil_Combined_Kspace_Phase[echo,:,:,:,freq]=Coil_Combined_Kspace_Phase[echo,:,:,:,freq]+np.angle(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(np.squeeze(Regridded_kspace[i,:,:,:,freq])))))**2
        			Regridded_kspace = np.zeros(shape=(int(NbCoils),Kx_size,Ky_size,Kz_size,L), dtype=np.complex64)		#Jacques edit test
                    
        			print ('   >> Projections = ', float(int(usedline)/int(NbProjections))*100, '%')
                    
        		Coil_Combined_temp[echo,:,:,:]=Coil_Combined_Kspace_Module[echo,:,:,:,freq]            
        		Coil_Combined_temp[echo,:,:,:]=np.sqrt((Coil_Combined_temp[echo,:,:,:]))
        		#### Image Reorientation to Scanner DICOM anatomical Images
        		## First we swap axes to bring the axial plane correctly opened in Anatomist compared to Anatomy
        		## Second we rotate of 180\B0 to fix the left/Right and Antero Posterior Orientation vs Anatomy
        		Coil_Combined_temp[echo,:,:,:]=np.swapaxes(Coil_Combined_temp[echo,:,:,:],0,2)
        		Coil_Combined_temp[echo,:,:,:]=np.swapaxes(Coil_Combined_temp[echo,:,:,:],0,1)
        		Coil_Combined_temp[echo,:,:,:]=np.rot90(Coil_Combined_temp[echo,:,:,:],2)            
        		Coil_Combined_Kspace_Module[echo,:,:,:,freq]=Coil_Combined_temp[echo,:,:,:]
        		print ('[done]')
                
                
	fh,fw,fl = np.shape(field_map)  
	final_image= np.zeros(shape=(echoes,fh,fw,fl))
    
	for echo in range (echoes):     		        
		if (recon_method=='fsc'):
			for x in range(fh):
				for y in range(fw):
        				for z in range(fl):
        					px_freq = field_map[x,y,z]*2*np.pi
        					idx=np.argmin(abs(deltaw-px_freq))
        					try:                            
        						final_image[echo,x,y,z] = Coil_Combined_Kspace_Module[echo,x,y,z,idx]
        					except:
        						print('hi')                        
    
		elif (recon_method=='mfi'):
			Coil_Combined_temp[echo,:,:,:]=Coil_Combined_Kspace_Module[echo,:,:,:]
			num_of_interp = 1000
			tk=64
			timesamples=np.linspace(0,Timesampling,tk)
			interpolation_omegas = np.linspace(np.min(field_map),np.max(field_map),num_of_interp)*2*np.pi
			deltawi_tk=np.exp(1j * np.transpose(timesamples) * deltaw)
			y= np.exp(1j*np.transpose(timesamples)*interpolation_omegas)
			coeff_table = np.transpose( np.inv(deltawi_tk) * y)
			Coil_Combined_Kspace_Module=np.swapaxes(Coil_Combined_Kspace_Module,2,3)
			Coil_Combined_Kspace_Module=np.swapaxes(Coil_Combined_Kspace_Module,3,4)
			final_image = np.zeros(fh,fw)
			for x in range(fw):
				for y in range(fh):
        				for z in range(fl):
        					px_omega=field_map[x,y,z]*2*np.pi
        					idx=np.argmin(abs(interpolation_omegas-px_omega))
        					#final_image[echo,x,y,z] = coeff_table[idx,:] * images[y,:,x]
        
	Abs_Sum_of_Regridded_kspace=np.absolute(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(np.squeeze(Sum_of_Regridded_kspace[:,:,:])))))**2
	if SaveKspace : return Coil_Combined_Kspace_Module, Coil_Combined_Kspace_Phase, Regridded_kspace
	else : return final_image, Coil_Combined_Kspace_Phase, Abs_Sum_of_Regridded_kspace
    
    
def DemodData(Data,deltaw,Timesampling,coilstart,NbCoils,echoes,NbProjections,NbPoints,OverSamplingFactor,sousech):
    
    Demodded_Data = np.zeros(shape=np.shape(Data), dtype=np.complex64)
    #deltaw = np.linspace(np.max(field_map), np.min(field_map), L)*2*np.pi
    time= np.linspace(0,Timesampling,NbPoints) 
    #dw_o_freq = deltaw[np.int(L/2)]    
    #num_of_interp=1000
    tk=64
    #timesamples=np.linspace(0,Timesampling,tk)
    #interpolation_omegas= np.linspace(np.min(field_map), np.max(field_map), num_of_interp)*2*np.pi
    #deltawi_tk=np.exp(1j * np.conj(timesamples * deltaw) )
    #y= np.exp(1j* timesamples.getH() * interpolation_omegas) 
    for echo in range (echoes):
    		
        print ('>> Griding echo ',echo)
        for i in range(coilstart,int(NbCoils)):
            print ('   >> regridding coil', i+1)
            usedline=0
            for l in range(NbProjections):
                # We generate a random value (Uniform Distribution (Gaussian ?)) and compare it with some threshold to remove the line
                rand= np.random.rand(1)  				
                if rand[0] > sousech : 
                    usedline+=1
                    # print (rand[0])
                    rand2= np.random.rand(1)
                    if rand2 >0 :
                        nbofpoints=NbPoints*OverSamplingFactor
                    else :
                        nbofpoints=NbPoints
                    for m in range(nbofpoints):
                        Demodded_Data[i][l][echo][m]= Data[i][l][echo][m]* np.exp(-1j * deltaw * time[m])
                        
    return(Demodded_Data)