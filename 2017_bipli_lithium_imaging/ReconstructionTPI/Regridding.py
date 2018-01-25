# -*- coding:Utf-8 -*-
# Author : Arthur Coste
# Date : December 2014
# Purpose : Regrid data  
#---------------------------------------------------------------------------------------------------------------------------------
from __future__ import print_function
import os
import argparse
import math,numpy,scipy.ndimage
from scipy.interpolate import griddata
from visualization import PlotImg,PlotImgMag,PlotReconstructedImage,PlotImg2,PlotImgMag2,DefineROIonImage
#from DataFilter import *

#from ctypes import *
#STD_OUTPUT_HANDLE_ID = c_ulong(0xfffffff5)
#windll.Kernel32.GetStdHandle.restype = c_ulong
#std_output_hdl = windll.Kernel32.GetStdHandle(STD_OUTPUT_HANDLE_ID)

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
	DataAvg=numpy.sum(Data[:,:,:,:,:],1) # sum over all averages
	
	if Data.size != 0 and NbLines and NbPoints and NbAverages and NbCoils and NbSlice and OverSamplingFactor:
		Average_Combined_Kspace = numpy.zeros(shape=(int(NbSlice),int(NbCoils),int(NbLines),int(NbPoints)*int(OverSamplingFactor)), dtype=numpy.complex64)
		Coil_Combined_Kspace = numpy.zeros(shape=(int(NbSlice),int(NbLines),int(NbPoints)*int(OverSamplingFactor)))
		PlotImgMag2(numpy.absolute(DataAvg[0][0][:][:]))
		# Begining of modifications for Jacques's data
		DataAvg[:,:,:,125:128]=0
		PlotImgMag2(numpy.absolute(DataAvg[0][0][:][:]))
		for i in range(NbSlice):
			DataAvg[i,0,0:16,:]=numpy.flipud(DataAvg[i,0,0:16,:])
		PlotImgMag2(numpy.absolute(DataAvg[3][0][:][:]))
		
		# temp=DataAvg[0,0,0:(NbLines/2),:];
		# DataAvg[0,0,0:(NbLines/2),:]=numpy.fliplr(numpy.conjugate(DataAvg[:,:,NbLines/2:NbLines,:]));
		# DataAvg[0,0,NbLines/2:NbLines,:]=numpy.fliplr(numpy.conjugate(numpy.flipud(temp)))
		# PlotImgMag2(numpy.absolute(DataAvg[0][0][:][:]))
		for j in range(NbSlice):
			for i in range(NbCoils):
				# Average_Combined_Kspace[j][i][:][:]=Average_Combined_Kspace[j][i][:][:]+ numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.fftshift(numpy.squeeze(DataAvg[j][i][:][:]))))
				Average_Combined_Kspace[j][i][:][:]=Average_Combined_Kspace[j][i][:][:]+ numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.fftshift(numpy.squeeze(DataAvg[j][i][:][:]))))
			Coil_Combined_Kspace[j][:][:]=Coil_Combined_Kspace[j][:][:]+numpy.absolute(Average_Combined_Kspace[j][i][:][:])*numpy.absolute(Average_Combined_Kspace[j][i][:][:])
			Coil_Combined_Kspace[j][:][:]=numpy.sqrt((Coil_Combined_Kspace[j][:][:]))
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
	DataAvg=numpy.sum(Data[:,:,:,:,:],1) # sum over all averages	
	print (DataAvg.shape)
	Coil_Combined_Kspace = numpy.zeros(shape=(int(NbSlice),int(NbLines),int(NbPoints)*int(OverSamplingFactor)))
	for i in range(NbSlice):
		for j in range(NbCoils):
				# for k in range(NbLines): 
					# Kspace[i][j][k][:] = Data[i][j][k][:]
			Coil_Combined_Kspace[:][:][:]+numpy.absolute(numpy.fft.fftshift(numpy.fft.ifftn(numpy.fft.fftshift(numpy.squeeze(DataAvg[i][j][:][:][:])))))**2
	print (Coil_Combined_Kspace.shape)
	PlotReconstructedImage((Coil_Combined_Kspace[4,:,:]))	
	# if Data.size != 0 and NbLines and NbPoints and NbAverages and NbCoils and NbSlice and OverSamplingFactor:
		# Average_Combined_Kspace = numpy.zeros(shape=(int(NbCoils),int(NbLines),int(NbPoints)*int(OverSamplingFactor)), dtype=numpy.complex64)
		# Coil_Combined_Kspace = numpy.zeros(shape=(int(NbLines),int(NbPoints)*int(OverSamplingFactor)))
		# for i in range(NbCoils):
			# for k in range(NbAverages):
				# Average_Combined_Kspace[i][:][:]=Average_Combined_Kspace[i][:][:]+ numpy.fft.fftshift(numpy.fft.ifftn(numpy.fft.fftshift(numpy.squeeze(Data[k][i][:][:]))))
			# Coil_Combined_Kspace[:][:]=Coil_Combined_Kspace[:][:]+numpy.absolute(Average_Combined_Kspace[i][:][:])*numpy.absolute(Average_Combined_Kspace[i][:][:])
		# Coil_Combined_Kspace[:][:]=numpy.sqrt((Coil_Combined_Kspace[:][:]))
		# PlotImgMag2((Coil_Combined_Kspace))
	return Coil_Combined_Kspace
	
def Radial_Regridding(NbLines,NbPoints,NbAverages,NbCoils,NbSlice,OverSamplingFactor,Nucleus,MagneticField,SubSampling,Data,Interpolation,verbose,Gmax=None,BW=None):

	print ('------------------------------------------------------------')
	print ('INFO    : Running 2D RADIAL Regridding Version 1.2')
	print ()
		
	if BW==None:
		try:
			#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 15)
			BW=float(raw_input('REQUEST : BandWidth per pixel = '))
			#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
		except ValueError:
			print ('Not a valid number')
	
	if Gmax==None:
		try:
			#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 15)
			Gmax=float(raw_input('REQUEST : ReadOut Gradient Amplitude = '))
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

		RO_GradShape = numpy.zeros(shape=(int(NbLines),int(NbPoints)*OverSamplingFactor))
		PE_GradShape = numpy.zeros(shape=(int(NbLines),int(NbPoints)*OverSamplingFactor))
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
		
		x = numpy.linspace(numpy.amin(RO_GradShape), numpy.amax(RO_GradShape),NbPoints*OverSamplingFactor*2)
		y = numpy.linspace(numpy.amin(PE_GradShape), numpy.amax(PE_GradShape),NbPoints*OverSamplingFactor*2)
		xv, yv = numpy.meshgrid(x, y)
		
		if verbose: print ('dims of ReadOut Gradient Matrix        = ',len(RO_GradShape),'x',len(RO_GradShape[0]))
		if verbose: print ('dims of Phase Encoding Gradient Matrix = ',len(PE_GradShape),'x',len(PE_GradShape[0]))
		if verbose: print ('dims of Output Grid X                  = ',len(xv),'x',len(xv[0]))
		if verbose: print ('dims of Output Grid Y                  = ',len(yv),'x',len(yv[0]))
		if verbose: print ('dims of Input values                   = ',len(Data[0][0][0][:]),'x',len(Data[0][0][0][0][:]))
		
		print("INFO    : Performing regridding",end="\t\t\t")
		
		Regridded_kspace = numpy.zeros(shape=(int(NbSlice),int(NbAverages),int(NbCoils),int(len(xv)),int(len(xv[0]))), dtype=numpy.complex64)		
		Average_Combined_Kspace = numpy.zeros(shape=(int(NbSlice),int(NbCoils),int(len(xv)),int(len(xv[0]))), dtype=numpy.complex64)
		Coil_Combined_Kspace = numpy.zeros(shape=(int(NbSlice),int(len(xv)),int(len(xv[0]))))
		
		RO_GradShape_V=numpy.zeros(RO_GradShape.size)
		RO_GradShape_V=numpy.reshape(RO_GradShape,RO_GradShape.size)
		PE_GradShape_V=numpy.zeros(PE_GradShape.size)
		PE_GradShape_V=numpy.reshape(PE_GradShape,PE_GradShape.size)
		xv_V=numpy.zeros(xv.size)
		xv_V=numpy.reshape(xv,xv.size)
		yv_V=numpy.zeros(yv.size)
		yv_V=numpy.reshape(yv,yv.size)
		
		for j in range(NbSlice):
			for i in range(int(NbCoils)):
				for k in range(NbAverages):
					Data_V=numpy.zeros(Data[j][k][i].size)
					Data_V=numpy.reshape(Data[j][k][i],Data[j][k][i].size)
					Regridded_kspace[j][k][i]=numpy.reshape(griddata((RO_GradShape_V,PE_GradShape_V),Data_V,(xv_V, yv_V),method=str(Interpolation),fill_value=0),(int(len(xv)),int(len(xv[0]))))
					Average_Combined_Kspace[j][i][:][:]=Average_Combined_Kspace[j][i][:][:]+ numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.fftshift(numpy.squeeze(Regridded_kspace[j][k][i][:][:]))))
					# PlotReconstructedImage(numpy.absolute(Average_Combined_Kspace[j][i]))
				Coil_Combined_Kspace[j][:][:]=Coil_Combined_Kspace[j][:][:]+numpy.absolute(Average_Combined_Kspace[j][i][:][:])**2
			Coil_Combined_Kspace[j][:][:]=numpy.sqrt((Coil_Combined_Kspace[j][:][:]))
			PlotReconstructedImage((Coil_Combined_Kspace[j]))
		PlotReconstructedImage(numpy.absolute((Regridded_kspace[0][0][0])))
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
		print (numpy.amin(KX), numpy.amax(KX))
		print (numpy.amin(KY), numpy.amax(KY))
		print (numpy.amin(KZ), numpy.amax(KZ))
		
		x = numpy.linspace(numpy.amin(KX), numpy.amax(KX),NbPoints*OverSamplingFactor/8)
		y = numpy.linspace(numpy.amin(KY), numpy.amax(KY),NbPoints*OverSamplingFactor/8)
		z = numpy.linspace(numpy.amin(KZ), numpy.amax(KZ),NbPoints*OverSamplingFactor/8)
		xv, yv, zv = numpy.meshgrid(x, y, z)
		
		Regridded_kspace = numpy.zeros(shape=(int(NbAverages),int(NbCoils),int(len(xv[0])),int(len(xv[0])),int(len(xv[0]))), dtype=numpy.complex64)		
		Average_Combined_Kspace = numpy.zeros(shape=(int(NbCoils),int(len(xv[0])),int(len(xv[0])),int(len(xv[0]))), dtype=numpy.complex64)
		Coil_Combined_Kspace = numpy.zeros(shape=(int(len(xv[0])),int(len(xv[0])),int(len(xv[0]))))
		
		X_GradShape_V=numpy.zeros(KX.size)
		X_GradShape_V=numpy.reshape(KX,KX.size)
		Y_GradShape_V=numpy.zeros(KY.size)
		Y_GradShape_V=numpy.reshape(KY,KY.size)
		Z_GradShape_V=numpy.zeros(KZ.size)
		Z_GradShape_V=numpy.reshape(KZ,KZ.size)
		xv_V=numpy.zeros(xv.size)
		xv_V=numpy.reshape(xv,xv.size)
		yv_V=numpy.zeros(yv.size)
		yv_V=numpy.reshape(yv,yv.size)
		zv_V=numpy.zeros(zv.size)
		zv_V=numpy.reshape(zv,zv.size)
		
		print (X_GradShape_V.shape, Y_GradShape_V.shape, Z_GradShape_V.shape)
		print (xv_V.shape, yv_V.shape, zv_V.shape)
		print (Data.shape)
		
		for i in range(NbCoils):
			for k in range(NbAverages):
				print ('starting regridding coil ',i)
				Regridded_kspace[k][i]=numpy.reshape(griddata((X_GradShape_V,Y_GradShape_V,Z_GradShape_V),Data,(xv_V, yv_V,zv_V),method=str(Interpolation),fill_value=0),(int(len(xv[0])),int(len(xv[0])),int(len(xv[0]))))
				# print('combining')
				Average_Combined_Kspace[i][:][:][:]=Average_Combined_Kspace[i][:][:][:]+ numpy.fft.fftshift(numpy.fft.ifftn(numpy.fft.fftshift(numpy.squeeze(Regridded_kspace[k][i][:][:][:]))))
			Coil_Combined_Kspace[:][:][:]=Coil_Combined_Kspace[:][:][:]+numpy.absolute(Average_Combined_Kspace[i][:][:][:])*numpy.absolute(Average_Combined_Kspace[i][:][:][:])
		Coil_Combined_Kspace[:][:][:]=numpy.sqrt((Coil_Combined_Kspace[:][:][:]))

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
	
	beta=numpy.pi*numpy.sqrt((w**2/a**2)*((a-0.5)**2)-0.8)		#Rapid Gridding Reconstruction With a Minimal Oversampling Ratio P. J. Beatty et al, 2005
	u=numpy.arange(0,length-1,dtype=float)/(length-1) * (int(w)/2)
	kernel=beta*numpy.sqrt(1-((2*u[numpy.absolute(u) < int(w)/2])/w)**2) 	#Selection of a Convolution Function for Fourier Inversion using Griding, Jackson et al 1991
	KBkernel=(numpy.i0(kernel))/width
	# KBkernel=KBkernel/KBkernel[0]
	print (KBkernel)
	
	return KBkernel, u,beta
	
def simple_poly_area(x, y):
    # For short arrays (less than about 100 elements) it seems that the
    # Python sum is faster than the numpy sum. Likewise for the Python
    # built-in abs.
    return .5 * abs(sum(x[:-1] * y[1:] - x[1:] * y[:-1]) +
                    x[-1] * y[0] - x[0] * y[-1])
  	
	
def KaiserBesselRegridding(NbLines,NbPoints,NbAverages,NbCoils,NbSlice,OverSamplingFactor,Nucleus,MagneticField,SubSampling,Data,Interpolation,verbose,Gmax=None,BW=None):

	if NbCoils==0 : NbCoils=1
	print (NbCoils)
	#Compute Kaiser Bessel
	NormalizedKernel, u, beta = CalculateKaiserBesselKernel(3,2,4)
	NormalizedKernelflip=numpy.flipud(NormalizedKernel)
	# print((NormalizedKernelflip))
	NormalizedKernelflip=numpy.delete(NormalizedKernelflip,2)
	NormalizedKernel=numpy.append(NormalizedKernelflip,NormalizedKernel)
	print((NormalizedKernel))
	NormalizedKernel=numpy.mat(NormalizedKernel)
	NormalizedKernel2D=numpy.transpose(NormalizedKernel)*(NormalizedKernel)	
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
			BW=float(raw_input('REQUEST : BandWidth per pixel = '))
			#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
		except ValueError:
			print ('Not a valid number')
	
	if Gmax==None:
		try:
			#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 15)
			Gmax=float(raw_input('REQUEST : ReadOut Gradient Amplitude = '))
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

	RO_GradShape = numpy.zeros(shape=(int(NbLines),int(NbPoints)*OverSamplingFactor))
	PE_GradShape = numpy.zeros(shape=(int(NbLines),int(NbPoints)*OverSamplingFactor))
	a=numpy.zeros(int(NbPoints)*OverSamplingFactor*int(NbLines))
	b=numpy.zeros(int(NbPoints)*OverSamplingFactor*int(NbLines))
	for i in range(NbLines):
		for j in range(NbPoints*OverSamplingFactor):
			# /!\ WARNING : I used a python vector for RO/PE_Amp... Maybe not that smart, but the first element is its size !
			if j< NbPts_On_Slope or j==NbPts_On_Slope:
				RO_GradShape[i][j]=Gamma*RO_Amp[i+1]/rampdur*(((j)*Tech)*((j)*Tech))/2
				PE_GradShape[i][j]=Gamma*PE_Amp[i+1]/rampdur*(((j)*Tech)*((j)*Tech))/2
				# a=numpy.append(a,RO_GradShape[i][j])
				# b=numpy.append(b,PE_GradShape[i][j])
				
			else:
				RO_GradShape[i][j]=Gamma*RO_Amp[i+1]*(((((NbPts_On_Slope)*(Tech))**2))/2/rampdur+(j-NbPts_On_Slope)*Tech)
				PE_GradShape[i][j]=Gamma*PE_Amp[i+1]*(((((NbPts_On_Slope)*(Tech))**2))/2/rampdur+(j-NbPts_On_Slope)*Tech)
				# a=numpy.append(a,RO_GradShape[i][j])
				# b=numpy.append(b,PE_GradShape[i][j])
	print("[DONE]")
	
	from scipy.spatial import Voronoi, voronoi_plot_2d
	import matplotlib.pyplot as plt

	# points = numpy.column_stack((a,b))
	points = numpy.column_stack((numpy.ravel(RO_GradShape),numpy.ravel(PE_GradShape)))
	del(a);del(b)
	# compute Voronoi tesselation
	vor = Voronoi(points)
	# voronoi_plot_2d(vor)
	# plt.show()
	# print (len(vor.regions))
	area = numpy.zeros(len(vor.regions))
	for node in range(len(vor.regions)):
		xv = vor.vertices[vor.regions[node],0]
		yv = vor.vertices[vor.regions[node],1]
		# print (xv,yv)
		if ((xv !=[] and yv !=[]) and (len(xv)==4)):
			area[node] = simple_poly_area(xv, yv)
			# print ("Aire ",node," == ",simple_poly_area(xv, yv))
	weightVoronoi=numpy.unique(numpy.round(area,9))
	weightVoronoi=numpy.sort(weightVoronoi)
	weightVoronoi=weightVoronoi+(1/(int(NbPoints)*OverSamplingFactor))
	weightVoronoi=weightVoronoi/max(weightVoronoi)
	weightVoronoi=numpy.append(weightVoronoi,1)
	weightVoronoi[0]=1e-5
	del(points);del(vor);del(area);del(xv);del(yv);

	# print (weightVoronoi)
	print (len(weightVoronoi))
	if not os.path.isfile("voronoi_coefs.txt") :
		numpy.savetxt("voronoi_coefs.txt",weightVoronoi)
	
	LinearWeigths=numpy.linspace(1/(NbPoints*OverSamplingFactor), 1,NbPoints*OverSamplingFactor)
	LinearWeigths=numpy.delete(LinearWeigths,0)
	LinearWeigths=numpy.append(LinearWeigths,1)
	# print (LinearWeigths)
	# print (len(LinearWeigths))
	if not os.path.isfile("linear_coefs.txt") :
		numpy.savetxt("linear_coefs.txt",LinearWeigths)

	# size = numpy.round((numpy.amax(PE_GradShape)-numpy.amin(PE_GradShape))/2)
	# size = NbPoints*OverSamplingFactor*2
	imsize = NbPoints*OverSamplingFactor*2
	# size = numpy.round(NbPoints*OverSamplingFactor*2*2)
	size = numpy.round(NbPoints*OverSamplingFactor*2*2)
	# On utilise une grille 2 fois plus grande pour minimiser les erreurs de regridding et on crop apr\E8s
	# size = NbPoints*OverSamplingFactor*2*2
	print(size,numpy.amax(PE_GradShape),numpy.amin(PE_GradShape))
	
	# Regridded_kspace = numpy.zeros(shape=(int(NbSlice),int(NbAverages),int(NbCoils),int(size),int(size)), dtype=numpy.complex64)		
	# Average_Combined_Kspace = numpy.zeros(shape=(int(NbSlice),int(NbCoils),int(size),int(size)), dtype=numpy.complex64)
	# Coil_Combined_Kspace = numpy.zeros(shape=(int(NbSlice),int(size),int(size)))
	
	# x_current=numpy.round((RO_GradShape/numpy.amax(RO_GradShape))*size/2)
	# y_current=numpy.round((PE_GradShape/numpy.amax(PE_GradShape))*size/2)
	print (Data.shape)
	DataAvg = numpy.zeros(shape=(int(NbSlice),int(NbCoils),int(size),int(size)), dtype=numpy.complex64)
	
	# IF CODE CRASHES THEN IT MEANS THAT DATA HAVE ALREADY BEEN SUMMED IN HIGHER FUNCTIONS
	DataAvg=numpy.sum(Data[:,:,:,:,:],1) # sum over all averages
	# DataAvg=numpy.sum(Data[:,1:2:,:,:],1)
	# DataAvg=numpy.mean(Data,1)
	print (DataAvg.shape)

	# Regridded_kspace = numpy.zeros(shape=(int(NbSlice),int(NbCoils),int(size)+2,int(size)+2), dtype=numpy.complex64)		
	# Coil_Combined_Kspace = numpy.zeros(shape=(int(NbSlice),int(size)+2,int(size)+2))
	Regridded_kspace = numpy.zeros(shape=(int(NbSlice),int(NbCoils),int(size),int(size)), dtype=numpy.complex64)		
	Coil_Combined_Kspace = numpy.zeros(shape=(int(NbSlice),int(size),int(size)), dtype=numpy.complex64)
	Coil_Combined_Kspace_Module = numpy.zeros(shape=(int(NbSlice),int(size),int(size)))
	Coil_Combined_Kspace_Phase = numpy.zeros(shape=(int(NbSlice),int(size),int(size)))
	usedline=0
	KspaceNRJ=0.0
	
	# N=20
	# chosen_golden_lines=numpy.zeros(N)
	# goldenAngle = 180/1.618;
	# for g in range(N):
		# chosen_golden_lines[g] = numpy.unique(numpy.mod(round(g*goldenAngle),360)) ;
	# print (chosen_golden_lines)
	
	for j in range (NbSlice):
		for i in range(int(NbCoils)):
			for l in range(NbLines):
				# We generate a random value (Uniform Distribution (Gaussian ?)) and compare it with some threshold to remove the line
				rand= numpy.random.rand(1)
				
				# current_angle = l*360/402;
				# if sum(round(current_angle)==chosen_golden_lines)==1:
				if rand[0] >= 0.0 : 
				# if numpy.mod(l,2)==0:
					usedline+=1
					rand2= numpy.random.rand(1)
					if rand2 >=0 :
						nbofpoints=NbPoints*OverSamplingFactor
					else :
						nbofpoints=NbPoints
					for m in range(nbofpoints):
						# Pour avoir le K space centr\E9
						# x_current=numpy.round(RO_GradShape[l][m]/numpy.amax(RO_GradShape)*size/2)-size/2+1
						# y_current=numpy.round(PE_GradShape[l][m]/numpy.amax(PE_GradShape)*size/2)-size/2+1
						x_current=numpy.round(RO_GradShape[l][m]/numpy.amax(RO_GradShape)*size/2)
						y_current=numpy.round(PE_GradShape[l][m]/numpy.amax(PE_GradShape)*size/2)
						# Val=DataAvg[j][i][l][m]
						Val=DataAvg[j][i][l][m]*weightVoronoi[m]
						# Val=Data[j][i][l][m]*LinearWeigths[m]
						# print(Val)
						KspaceNRJ+=(float(numpy.absolute(Val)**2)*(1/float(BW)))
						
						for a in range(-1,1,1):
							for b in range(-1,1,1):
								# Regridded_kspace[j][i][y_current[l,m]+a][x_current[l,m]+b]=Regridded_kspace[j][i][y_current[l,m]+a][x_current[l,m]+b]+Val*NormalizedKernel2D[a+1,b+1]
								Regridded_kspace[j][i][y_current+a][x_current+b]=Regridded_kspace[j][i][y_current+a][x_current+b]+Val*NormalizedKernel2D[a+1,b+1]*numpy.exp(-i*(numpy.pi/4)*1j)
								# Regridded_kspace[j][i][-(y_current+a)][-(x_current+b)]=numpy.conjugate(Regridded_kspace[j][i][y_current+a][x_current+b])
								# Regridded_kspace[j][i][y_current+a][x_current+b]=Regridded_kspace[j][i][y_current+a][x_current+b]+Val*NormalizedKernel2D[a+1,b+1]
						# Regridded_kspace[j][i][y_current][x_current]=Regridded_kspace[j][i][y_current][x_current]+Val
						# Regridded_kspace[j][i][y_current[l,m]][x_current[l,m]]=Regridded_kspace[j][i][y_current[l,m]][x_current[l,m]]+Val
			Coil_Combined_Kspace[j]=numpy.sum(Regridded_kspace,1)
			Coil_Combined_Kspace_Module[j][:][:]=Coil_Combined_Kspace_Module[j][:][:]+numpy.absolute(numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.fftshift(numpy.squeeze(Regridded_kspace[j][i][:][:])))))**2
			# Coil_Combined_Kspace_Phase[j][:][:]=Coil_Combined_Kspace_Phase[j][:][:]+numpy.angle(numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.fftshift(numpy.squeeze(Regridded_kspace[j][i][:][:])))))
		Coil_Combined_Kspace_Module[j][:][:]=numpy.sqrt((Coil_Combined_Kspace_Module[j][:][:]))
		
		Coil_Combined_Kspace_Phase[j]=numpy.angle(numpy.fft.fftshift(numpy.fft.ifft2((numpy.squeeze(Coil_Combined_Kspace[j])))))
		PlotReconstructedImage(Coil_Combined_Kspace_Phase[j])
		# C = InverseFunc(len(Coil_Combined_Kspace_Module[0]),beta,3)
		# C = InverseFunc(len(Coil_Combined_Kspace_Module[0])/2,beta,3)
		# PlotReconstructedImage(Coil_Combined_Kspace_Module[j,imsize/2:imsize+imsize/2,imsize/2:imsize+imsize/2])	
		PlotReconstructedImage(Coil_Combined_Kspace_Module[j,numpy.round((size-imsize)/2):imsize+numpy.round((size-imsize)/2),numpy.round((size-imsize)/2):imsize+numpy.round((size-imsize)/2)])	
		PlotReconstructedImage(Coil_Combined_Kspace_Module[j])	
		# PlotReconstructedImage((Coil_Combined_Kspace_Module[j,imsize/2:imsize+imsize/2,imsize/2:imsize+imsize/2])/C)
	
	PlotImgMag(numpy.absolute((Regridded_kspace[0][0])))
	print("Reconstructed with ", float((float(usedline)/float(NbLines*NbCoils))*100), " % of acquired lines")
	from processingFunctions import ComputeFullImageSNR
	ImageSNR = ComputeFullImageSNR(Coil_Combined_Kspace_Module[j])
	print ("Estimated SNR of Image = ", ImageSNR, "dB")
	print ("K space NRJ = ", KspaceNRJ)
	print ("checksum = ",sum(sum(sum(Coil_Combined_Kspace_Module))))
	print("[DONE]")
	
	# return Coil_Combined_Kspace_Module[j][size / 4: -size / 4, size / 4: - size / 4]
	# return Coil_Combined_Kspace_Module[j,imsize/2:imsize+imsize/2,imsize/2:imsize+imsize/2]
	return Coil_Combined_Kspace_Module[j,numpy.round((size-imsize)/2):imsize+numpy.round((size-imsize)/2),numpy.round((size-imsize)/2):imsize+numpy.round((size-imsize)/2)],Coil_Combined_Kspace_Phase[j,numpy.round((size-imsize)/2):imsize+numpy.round((size-imsize)/2),numpy.round((size-imsize)/2):imsize+numpy.round((size-imsize)/2)]
	# return Coil_Combined_Kspace_Module,Coil_Combined_Kspace_Phase[j,numpy.round((size-imsize)/2):imsize+numpy.round((size-imsize)/2),numpy.round((size-imsize)/2):imsize+numpy.round((size-imsize)/2)]
	
def InverseFunc(size,beta,width):
	C = numpy.zeros(shape=(size,size))
	A= numpy.zeros(size)

	print(beta)
	print (size)
				
	for i in range (-int(round(size/2)),int(round(size/2)),1):
			if i !=0:
				A[i]=(numpy.sin(numpy.sqrt((numpy.pi**2)*(width**2)*((i)**2)-beta**2))/(numpy.sqrt((numpy.pi**2)*(width**2)*((i)**2)-beta**2)))
			if i ==0: 
				A[i]=numpy.mean([A[int(round(size/2))-2],A[int(round(size/2))-1],A[int(round(size/2))+1],A[int(round(size/2))+2]])	
	
	# for i in range (int(size)):
		# if (i-size/2) !=0:
			# A[i-size/2]=(numpy.sin(numpy.sqrt((numpy.pi**2)*(width**2)*((i-size/2)**2)-(beta)**2)))/(numpy.sqrt((numpy.pi**2)*(width**2)*((i-size/2)**2)-(beta)**2))
		# if (i-size/2) ==0: 
			# A[i]=1e-6
	# for i in range (-int(round(size/2)),int(round(size/2)),1):
		# if i !=0:
			# A[i-int(round(size/2))]=(numpy.sin(numpy.sqrt((numpy.pi**2)*(width**2)*((i)**2)-beta**2))/(numpy.sqrt((numpy.pi**2)*(width**2)*((i)**2)-beta**2)))
	# A[int(round(size/2))]=numpy.mean([A[-2+int(round(size/2))],A[-1+int(round(size/2))],A[1+int(round(size/2))],A[2+int(round(size/2))]])		
	A[0]=1
	A=numpy.absolute(A)
	# Af=numpy.flipud(A)
	# Af=numpy.delete(Af,int(size/2)-1)
	# A=numpy.append(Af,A)
	# print(A)
	# A=A/numpy.max(A)
	A=numpy.mat(A)
	C=numpy.transpose(A)*(A)		
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
	# NormalizedKernelflip=numpy.flipud(NormalizedKernel)
	# print((NormalizedKernelflip))
	# NormalizedKernelflip=numpy.delete(NormalizedKernelflip,2)
	# NormalizedKernel=numpy.append(NormalizedKernelflip,NormalizedKernel)
	# print((NormalizedKernel))
	# NormalizedKernel=numpy.mat(NormalizedKernel)
	# NormalizedKernel2D=numpy.transpose(NormalizedKernel)*(NormalizedKernel)	
	# print(NormalizedKernel2D)
	# print(NormalizedKernel2D.shape)

	size = NbPoints*2

	Regridded_kspace = numpy.zeros(shape=(int(NbCoils),int(size),int(size),int(size)), dtype=numpy.complex64)		
	Coil_Combined_Kspace = numpy.zeros(shape=(int(size),int(size),int(size)))

	print (numpy.amin(Kx)*(size/1),numpy.amax(Kx)*(size/1))
	print (numpy.amin(Ky)*(size/1),numpy.amax(Ky)*(size/1))
	print (numpy.amin(Kz)*(size/1),numpy.amax(Kz)*(size/1))
	print (len(Kx))
	print (len(Ky))
	print (len(Kz))
	
	Kxloc=numpy.zeros(Kx.size);Kyloc=numpy.zeros(Ky.size);Kzloc=numpy.zeros(Kz.size);
	Kxloc=numpy.round(((Kx/numpy.amax(Kx))*((size-2)/2))-size/2)
	Kyloc=numpy.round(((Ky/numpy.amax(Ky))*((size-2)/2))-size/2)
	Kzloc=numpy.round(((Kz/numpy.amax(Kz))*((size-2)/2))-size/2)
	usedline=0
	for i in range(int(NbCoils)):
		for l in range(NbLines):
			if l%1000 ==0 : print (l)
			rand= numpy.random.rand(1)
			if rand[0] > 0.0 : 
				usedline+=1
				rand2= numpy.random.rand(1)
				if rand2 >0.0 :
					nbofpoints=NbPoints
				else :
					nbofpoints=NbPoints/2
				for m in range(nbofpoints):
					# x_current=int((numpy.round(Kx[(l*NbPoints)+m]/numpy.amax(Kx))*size/2)-size/2)
					# y_current=int((numpy.round(Ky[(l*NbPoints)+m]/numpy.amax(Ky))*size/2)-size/2)
					# z_current=int((numpy.round(Kz[(l*NbPoints)+m]/numpy.amax(Kz))*size/2)-size/2)
					# Val=Data[l][m]
					Val=Data[l][m]
					# for c in range(-1,1,1):
						# for a in range(-1,1,1):
							# for b in range(-1,1,1):
								# Regridded_kspace[k][i][z_current+c][y_current+a][x_current+b]=Regridded_kspace[k][i][z_current+c][y_current+a][x_current+b]+Val*NormalizedKernel2D[a+2,b+2]
					# Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]=Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]+Val
					Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]=Val
					# Regridded_kspace[i][-Kzloc[(l*NbPoints)+m]][-Kyloc[(l*NbPoints)+m]][-Kxloc[(l*NbPoints)+m]]=numpy.conj(Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]])
		# PlotImgMag(numpy.absolute((Regridded_kspace[0][0][48])))
		# Average_Combined_Kspace[i][:][:][:]=Average_Combined_Kspace[i][:][:][:]+ numpy.fft.fftshift(numpy.fft.ifftn(numpy.fft.fftshift(numpy.squeeze(Regridded_kspace[k][i][:][:][:]))))
		Coil_Combined_Kspace[:][:][:]=Coil_Combined_Kspace[:][:][:]+numpy.absolute(numpy.fft.fftshift(numpy.fft.ifftn(numpy.fft.ifftshift(numpy.squeeze(Regridded_kspace[i][:][:][:])))))**2
	Coil_Combined_Kspace[:][:][:]=numpy.sqrt((Coil_Combined_Kspace[:][:][:]))
	
	PlotReconstructedImage(Coil_Combined_Kspace[:][48][:])
	print (Regridded_kspace[0].shape)
	print("Reconstructed with ", float((float(usedline)/float(NbLines*NbCoils))*100), " % of acquired lines")
	
	return Coil_Combined_Kspace,numpy.absolute(Regridded_kspace[0]),numpy.angle(Regridded_kspace[0])
	
def KaiserBesselTPI(NbProjections,NbPoints,NbAverages,NbCoils,OverSamplingFactor,Nucleus,MagneticField,SubSampling,Data,KX,KY,KZ,pval,resolution,FOV,verbose,PSF_ND,PSF_D,B1sensitivity):
	NormalizedKernel, u, beta = CalculateKaiserBesselKernel(3,2,4)
	NormalizedKernelflip=numpy.flipud(NormalizedKernel)
	# print((NormalizedKernelflip))
	NormalizedKernelflip=numpy.delete(NormalizedKernelflip,2)
	NormalizedKernel=numpy.append(NormalizedKernelflip,NormalizedKernel)
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
	if KX!=None : 
		print("INFO    : KX [OK]")
		print('INFO    : KX bounds : ',numpy.amin(KX), numpy.amax(KX))
	else : print ("ERROR    :  [KX]") 
	if KY!=None : 
		print("INFO    : KY [OK]")
		print('INFO    : KY bounds : ',numpy.amin(KY), numpy.amax(KY))
	else : print ("ERROR    :  [KY]") 
	if KZ!=None : 
		print("INFO    : KZ [OK]")
		print('INFO    : KZ bounds : ',numpy.amin(KZ), numpy.amax(KZ))
	else : print ("ERROR    :  [KZ]") 
	
	
	if PSF_D: 
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 15)
		decay=float(raw_input('REQUEST :  T2/T2* decay [us] >> '))
		ReadOutTime=float(raw_input('REQUEST :  ReadOutTime [us] >> '))
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
		decroissance = numpy.zeros(NbPoints)
		for i in range(NbPoints):
			decroissance[i]=numpy.exp(-(i*ReadOutTime/NbPoints)/decay)
	
	NbCoils=int(NbCoils+1)
	# BW=260.0
	# size = NbPoints*OverSamplingFactor*2 # In this case we have twice the number of points (each line covers half of the plan)
	# On utilise une grille 2 fois plus grande pour minimiser les erreurs de regridding et on crop apr\E8s
	# size = NbPoints*OverSamplingFactor*2*2
	print (NbProjections)
	print(size)
	print (Data.shape)
	# DataAvg = numpy.zeros(shape=(int(NbCoils),int(NbProjections),int(size)), dtype=numpy.complex64)
	# DataAvg=numpy.sum(Data,0)
	# DataAvg=numpy.mean(Data,0)
	# print (DataAvg.shape)
	# size=size/16
	# size=numpy.round(size/4)
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
	
	linweights=numpy.ones(NbPoints,dtype=float)
	print (linweights.shape)
	# for i in range(int(numpy.round(len(linweights)*float(pval)))):
	for i in range(len(linweights)/2):
		# linweights[i]=float(i)*float(1/(numpy.round(float(len(linweights))*float(pval))))+1e-04
		linweights[i]=numpy.power(float(i)*float(1/(numpy.round(float(len(linweights))*float(pval)))),1.25)+1e-04
		# linweights[i]=float(i**2)*float(1/96.0)+1e-04
		# print (linweights[i])
		
	Regridded_kspace = numpy.zeros(shape=(int(NbCoils),Kx_size,Ky_size,Kz_size), dtype=numpy.complex64)		
	Coil_Combined_Kspace = numpy.zeros(shape=(Kx_size,Ky_size,Kz_size))
	# Regridded_kspace = numpy.zeros(shape=(int(NbCoils),int(size),int(size),int(size)), dtype=numpy.complex64)		
	# Coil_Combined_Kspace = numpy.zeros(shape=(int(size),int(size),int(size)))
	usedline=0
	if (B1sensitivity): 
		sousech=0.95
	else : sousech=0.0
	# KspaceNRJ=0.0
	
	KX = numpy.ravel(KX); KY=numpy.ravel(KY); KZ=numpy.ravel(KZ)
	Kxloc=numpy.zeros(KX.size);Kyloc=numpy.zeros(KX.size);Kzloc=numpy.zeros(KX.size);
	# Kxloc=numpy.round(((KX/numpy.amax(KX))*((size)))-(size)-2)
	# Kyloc=numpy.round(((KY/numpy.amax(KY))*((size)/2))-(size/2)-2)
	# Kzloc=numpy.round(((KZ/numpy.amax(KZ))*((size)/8))-(size/8)-2)
	Kxloc=numpy.floor(((KX/numpy.amax(KX))*((size/2))-(size/2)+1))
	Kyloc=numpy.floor(((KY/numpy.amax(KY))*((size/2))-(size/2)+1))
	Kzloc=numpy.floor(((KZ/numpy.amax(KZ))*((size/2))-(size/2)+1))
	
	DensityCompensationCoefficients = numpy.zeros(NbPoints,dtype=float)
	for i in range (len(DensityCompensationCoefficients)):
		DensityCompensationCoefficients[i]=numpy.sqrt((KX[i+1]-KX[i])**2 + (KY[i+1]-KY[i])**2 + (KZ[i+1]-KZ[i])**2)*numpy.sqrt((KX[i]/numpy.amax(KX))**2 + (KY[i]/numpy.amax(KY))**2 + (KZ[i]/numpy.amax(KZ))**2)**2
		# print("2  :",numpy.sqrt((KX[i+1]-KX[i])**2 + (KY[i+1]-KY[i])**2 + (KZ[i+1]-KZ[i])**2))
	

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
	
	# Data=numpy.sum(Data[:,:,:,:],0)
	print (Data.shape)
	
	for i in range(coilstart,int(NbCoils)):
		for l in range(NbProjections):
			# We generate a random value (Uniform Distribution (Gaussian ?)) and compare it with some threshold to remove the line
			rand= numpy.random.rand(1)
			
			if rand[0] > sousech : 
				usedline+=1
				# print (rand[0])
				rand2= numpy.random.rand(1)
				if rand2 >0 :
					nbofpoints=NbPoints*OverSamplingFactor
				else :
					nbofpoints=NbPoints
				for m in range(nbofpoints):
					# Pour avoir le K space centr\E9
					# x_current=numpy.round(KX[l][m]/numpy.amax(KX)*size/2)-size/2-1
					# y_current=numpy.round(KY[l][m]/numpy.amax(KY)*size/2)-size/2-1
					# z_current=numpy.round(KZ[l][m]/numpy.amax(KZ)*size/2)-size/2-1
					# Val=DataAvg[i][l][m]
					if (PSF_ND and not PSF_D) : Val=1
					if (PSF_D and not PSF_ND)  : Val=decroissance[m]
					elif (not PSF_ND and not PSF_D) : Val=Data[i][l][m]*DensityCompensationCoefficients[m]
					# print (Val)
					# Val=DataAvg[j][i][l][m]*weightVoronoi[m]
					# Val=Data[j][i][l][m]*LinearWeigths[m]
					# print(Val)
					# KspaceNRJ+=(float(numpy.absolute(Val)**2)*(1/float(BW)))
					
					for a in range(-1,1,1):
						for b in range(-1,1,1):
							for c in range(-1,1,1):
								Regridded_kspace[i][Kzloc[(l*NbPoints)+m+a]][Kyloc[(l*NbPoints)+m+b]][Kxloc[(l*NbPoints)+m+c]]=Regridded_kspace[i][Kzloc[(l*NbPoints)+m+a]][Kyloc[(l*NbPoints)+m+b]][Kxloc[(l*NbPoints)+m+c]]+Val*NormalizedKernel[c+1]
					# Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]=Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]+Val
		Coil_Combined_Kspace[:][:][:]=Coil_Combined_Kspace[:][:][:]+numpy.absolute(numpy.fft.fftshift(numpy.fft.ifftn(numpy.fft.fftshift(numpy.squeeze(Regridded_kspace[i][:][:])))))**2
		print ('regridding coil', i+1)
		print ('used lines = ', usedline)
	Coil_Combined_Kspace[:][:][:]=numpy.sqrt((Coil_Combined_Kspace[:][:][:]))
	
	# PlotImgMag(numpy.absolute((Regridded_kspace[4][:][:])))
	# return Regridded_kspace[0]
	return Coil_Combined_Kspace
	
def KaiserBesselTPI_ME(NbProjections,NbPoints,NbAverages,NbCoils,OverSamplingFactor,Nucleus,MagneticField,SubSampling,Data,KX,KY,KZ,pval,resolution,FOV,verbose,PSF_ND,PSF_D,B1sensitivity,echoes,SaveKspace):
	NormalizedKernel, u, beta = CalculateKaiserBesselKernel(3,2,4)
	NormalizedKernelflip=numpy.flipud(NormalizedKernel)
	# print((NormalizedKernelflip))
	NormalizedKernelflip=numpy.delete(NormalizedKernelflip,2)
	NormalizedKernel=numpy.append(NormalizedKernelflip,NormalizedKernel)
	NormalizedKernel = NormalizedKernel/numpy.amax(NormalizedKernel)
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
	if KX!=None : 
		print("INFO    : KX [OK]")
		print('INFO    : KX bounds : ',numpy.amin(KX), numpy.amax(KX))
	else : print ("ERROR    :  [KX]") 
	if KY!=None : 
		print("INFO    : KY [OK]")
		print('INFO    : KY bounds : ',numpy.amin(KY), numpy.amax(KY))
	else : print ("ERROR    :  [KY]") 
	if KZ!=None : 
		print("INFO    : KZ [OK]")
		print('INFO    : KZ bounds : ',numpy.amin(KZ), numpy.amax(KZ))
	else : print ("ERROR    :  [KZ]") 
	
	
	if PSF_D: 
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 15)
		decay=float(raw_input('REQUEST :  T2/T2* decay [us] >> '))
		ReadOutTime=float(raw_input('REQUEST :  ReadOutTime [us] >> '))
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
		decroissance = numpy.zeros(NbPoints)
		for i in range(NbPoints):
			decroissance[i]=numpy.exp(-(i*ReadOutTime/NbPoints)/decay)
	
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
	
	linweights=numpy.ones(NbPoints,dtype=float)
	# print (linweights.shape)
	# for i in range(int(numpy.round(len(linweights)*float(pval)))):
	for i in range(len(linweights)/2):
		# linweights[i]=float(i)*float(1/(numpy.round(float(len(linweights))*float(pval))))+1e-04
		linweights[i]=numpy.power(float(i)*float(1/(numpy.round(float(len(linweights))*float(pval)))),1.25)+1e-04
		# linweights[i]=float(i**2)*float(1/96.0)+1e-04
		# print (linweights[i])
		
	Regridded_kspace = numpy.zeros(shape=(int(NbCoils),Kx_size,Ky_size,Kz_size), dtype=numpy.complex64)		
	Sum_of_Regridded_kspace = numpy.zeros(shape=(int(NbCoils),Kx_size,Ky_size,Kz_size), dtype=numpy.complex64)		
	Coil_Combined_Kspace_Module = numpy.zeros(shape=(echoes,Kx_size,Ky_size,Kz_size))
	Coil_Combined_Kspace_Phase = numpy.zeros(shape=(echoes,Kx_size,Ky_size,Kz_size))
	Abs_Sum_of_Regridded_kspace = numpy.zeros(shape=(Kx_size,Ky_size,Kz_size))
	# Regridded_kspace = numpy.zeros(shape=(int(NbCoils),int(size),int(size),int(size)), dtype=numpy.complex64)		
	# Coil_Combined_Kspace = numpy.zeros(shape=(int(size),int(size),int(size)))
	
	if (B1sensitivity): 
		sousech=0.95
	else : sousech=0.0
	# KspaceNRJ=0.0
	
	KX = numpy.ravel(KX); KY=numpy.ravel(KY); KZ=numpy.ravel(KZ)
	Kxloc=numpy.zeros(KX.size);Kyloc=numpy.zeros(KX.size);Kzloc=numpy.zeros(KX.size);
	# Kxloc=numpy.round(((KX/numpy.amax(KX))*((size)))-(size)-2)
	# Kyloc=numpy.round(((KY/numpy.amax(KY))*((size)/2))-(size/2)-2)
	# Kzloc=numpy.round(((KZ/numpy.amax(KZ))*((size)/8))-(size/8)-2)
	Kxloc=numpy.floor(((KX/numpy.amax(KX))*((size/2))-(size/2)+1))
	Kyloc=numpy.floor(((KY/numpy.amax(KY))*((size/2))-(size/2)+1))
	Kzloc=numpy.floor(((KZ/numpy.amax(KZ))*((size/2))-(size/2)+1))
	
	DensityCompensationCoefficients = numpy.zeros(NbPoints,dtype=float)
	for i in range (len(DensityCompensationCoefficients)):
		DensityCompensationCoefficients[i]=numpy.sqrt((KX[i+1]-KX[i])**2 + (KY[i+1]-KY[i])**2 + (KZ[i+1]-KZ[i])**2)*numpy.sqrt((KX[i]/numpy.amax(KX))**2 + (KY[i]/numpy.amax(KY))**2 + (KZ[i]/numpy.amax(KZ))**2)**2		
	
	DensityCompensationCoefficients[int(NbPoints-1)]=DensityCompensationCoefficients[int(NbPoints-2)]
	DensityCompensationCoefficients[0]=DensityCompensationCoefficients[3]
	DensityCompensationCoefficients[1]=DensityCompensationCoefficients[3]
	DensityCompensationCoefficients[2]=DensityCompensationCoefficients[3]
	
	# DensityCompensationCoefficients = numpy.zeros(shape=(NbProjections,NbPoints),dtype=float)
	# for p in range ((NbProjections)):
		# for i in range (NbPoints):
			# DensityCompensationCoefficients[p,i]=numpy.sqrt((KX[p*NbPoints+i+1]-KX[p*NbPoints+i])**2 + (KY[p*NbPoints+i+1]-KY[p*NbPoints+i])**2 + (KZ[p*NbPoints+i+1]-KZ[p*NbPoints+i])**2)*numpy.sqrt((KX[p*NbPoints+i]/numpy.amax(KX))**2 + (KY[p*NbPoints+i]/numpy.amax(KY))**2 + (KZ[p*NbPoints+i]/numpy.amax(KZ))**2)**2		
	
		# DensityCompensationCoefficients[p,int(NbPoints-1)]=DensityCompensationCoefficients[p,int(NbPoints-2)]
		# DensityCompensationCoefficients[p,0]=DensityCompensationCoefficients[p,3]
		# DensityCompensationCoefficients[p,1]=DensityCompensationCoefficients[p,3]
		# DensityCompensationCoefficients[p,2]=DensityCompensationCoefficients[p,3]
	
	# print (DensityCompensationCoefficients.shape)
	
	# Data=numpy.sum(Data[:,:,:,:],0)
	# print (Data.shape)
	if (len(Data.shape)) >4:
		Data=numpy.sum(Data[:,:,:,:,:],0)
	print (Data.shape)
	
	for echo in range (echoes):
		
		print ('>> Griding echo ',echo)
		for i in range(coilstart,int(NbCoils)):
			print ('   >> regridding coil', i+1)
			usedline=0
			for l in range(NbProjections):
				# We generate a random value (Uniform Distribution (Gaussian ?)) and compare it with some threshold to remove the line
				rand= numpy.random.rand(1)
				
				if rand[0] > sousech : 
					usedline+=1
					# print (rand[0])
					rand2= numpy.random.rand(1)
					if rand2 >0 :
						nbofpoints=NbPoints*OverSamplingFactor
					else :
						nbofpoints=NbPoints
					for m in range(nbofpoints):
						# Pour avoir le K space centr\E9
						# x_current=numpy.round(KX[l][m]/numpy.amax(KX)*size/2)-size/2-1
						# y_current=numpy.round(KY[l][m]/numpy.amax(KY)*size/2)-size/2-1
						# z_current=numpy.round(KZ[l][m]/numpy.amax(KZ)*size/2)-size/2-1
						# Val=DataAvg[i][l][m]
						if (PSF_ND and not PSF_D) : Val=1
						if (PSF_D and not PSF_ND)  : Val=decroissance[m]
						elif (not PSF_ND and not PSF_D) : Val=Data[i][l][echo][m]*DensityCompensationCoefficients[m]
						# print (numpy.squeeze(Data[i][l][echo][m]))
						# print (Val)
						# Val=DataAvg[j][i][l][m]*weightVoronoi[m]
						# Val=Data[j][i][l][m]*LinearWeigths[m]
						# print(Val)
						# KspaceNRJ+=(float(numpy.absolute(Val)**2)*(1/float(BW)))
						
						for a in range(-1,1,1):
							for b in range(-1,1,1):
								for c in range(-1,1,1):
									# print (Regridded_kspace[i][Kzloc[(l*NbPoints)+m+a]][Kyloc[(l*NbPoints)+m+b]][Kxloc[(l*NbPoints)+m+c]])
									# print (Val)
									Regridded_kspace[i][Kzloc[(l*NbPoints)+m+a]][Kyloc[(l*NbPoints)+m+b]][Kxloc[(l*NbPoints)+m+c]]=Regridded_kspace[i][Kzloc[(l*NbPoints)+m+a]][Kyloc[(l*NbPoints)+m+b]][Kxloc[(l*NbPoints)+m+c]]+Val*NormalizedKernel[c+1]
						# Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]=Regridded_kspace[i][Kzloc[(l*NbPoints)+m]][Kyloc[(l*NbPoints)+m]][Kxloc[(l*NbPoints)+m]]+Val
			Sum_of_Regridded_kspace[:,:,:] = Sum_of_Regridded_kspace[:,:,:] + Regridded_kspace[:,:,:]		
			Coil_Combined_Kspace_Module[echo,:,:,:]=Coil_Combined_Kspace_Module[echo,:,:,:]+numpy.absolute(numpy.fft.fftshift(numpy.fft.ifftn(numpy.fft.fftshift(numpy.squeeze(Regridded_kspace[i][:][:])))))**2
			Coil_Combined_Kspace_Phase[echo,:,:,:]=Coil_Combined_Kspace_Phase[echo,:,:,:]+numpy.angle(numpy.fft.fftshift(numpy.fft.ifftn(numpy.fft.fftshift(numpy.squeeze(Regridded_kspace[i][:][:])))))**2
			
			print ('   >> Projections = ', float(int(usedline)/int(NbProjections))*100, '%')
		Coil_Combined_Kspace_Module[echo,:,:,:]=numpy.sqrt((Coil_Combined_Kspace_Module[echo,:,:,:]))
		#### Image Reorientation to Scanner DICOM anatomical Images
		## First we swap axes to bring the axial plane correctly opened in Anatomist compared to Anatomy
		## Second we rotate of 180\B0 to fix the left/Right and Antero Posterior Orientation vs Anatomy
		Coil_Combined_Kspace_Module[echo,:,:,:]=numpy.swapaxes(Coil_Combined_Kspace_Module[echo,:,:,:],0,2)
		Coil_Combined_Kspace_Module[echo,:,:,:]=numpy.swapaxes(Coil_Combined_Kspace_Module[echo,:,:,:],0,1)
		Coil_Combined_Kspace_Module[echo,:,:,:]=numpy.rot90(Coil_Combined_Kspace_Module[echo,:,:,:],2)
	print ('[done]')
	# PlotImgMag(numpy.absolute((Regridded_kspace[4][:][:])))
	# return Regridded_kspace[0]
	Abs_Sum_of_Regridded_kspace=numpy.absolute(numpy.fft.fftshift(numpy.fft.ifftn(numpy.fft.fftshift(numpy.squeeze(Sum_of_Regridded_kspace[:,:,:])))))**2
	if SaveKspace : return Coil_Combined_Kspace_Module, Coil_Combined_Kspace_Phase, Regridded_kspace
	else : return Coil_Combined_Kspace_Module, Coil_Combined_Kspace_Phase, Abs_Sum_of_Regridded_kspace
