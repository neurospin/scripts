# -*- coding:Utf-8 -*-
#
# Author : Arthur Coste
# Date : December 2014
# Purpose : Process Siemens Raw Data File 
#			Return MultiDimensional array containing K space Complex Data
#			Return Acquisition Parameters (NbLines,NbPoints,NbAverages,NbCoils,NbSlice,OverSamplingFactor,Nucleus,MagneticField)
#---------------------------------------------------------------------------------------------------------------------------------

# Importing modules and libraries
import os
import re, math, struct
import numpy as np
from visualization import PlotImg
import binascii

#from ctypes import windll,c_ulong
#STD_OUTPUT_HANDLE_ID = c_ulong(0xfffffff5)
#windll.Kernel32.GetStdHandle.restype = c_ulong
#std_output_hdl = windll.Kernel32.GetStdHandle(STD_OUTPUT_HANDLE_ID)

def ExtractDataFromRawData(source_file,verbose,viz,HeaderOnly):
	# READ RAWDATA FROM MRI ACQUISITION ON SIEMENS MACHINE
	
	# color 10=vert 12=rouge
	#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
	print "------------------------------------------------------------"
	print "INFO    : Running SiRaw Version 1.4"		# SIemens Raw
	print
	
	if viz: 
		visualization=True
		print 'INFO    : Visualization is Enabled'
	else : 
		print 'INFO    : Vizualization is Disabled !'
		visualization=False

	cplx=[]
	trajectory=[]
	# An update in sequence code on January 15th, 2015 allows to return ROGradmax and Bandwith per pixel
	# Those fields don't exist in previous rawdata file so we initialize them to empty values
	ROGradmax=[];BandWidth=[];nbLines_ICEOFF=[];nbLines=[];nbPoints=[];nbSlices=[];averages=[];coils=[];nbPartitions=[];Orientation=[];
	oversamplingFactor=[];Nucleus=[];MagneticField=[];SliceThickness=[];ReadoutFOV=[];PhaseFOV=[];FlipAngle=[];TE=[];TR=[];ACQTime=[];
	with open(source_file, 'rb') as source:
		if verbose: print 'INFO    : Extracting acquisition parameters'
		header_size=source.read(4)
		header_size=struct.unpack('I', header_size)
		if verbose: print 'INFO    : Header size = ',int(header_size[0])
		
		for line in source: 
		
			# pattern1 = '<ParamLong."BaseResolution">  {*'
			# rx1 = re.compile(pattern1, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern1 = 'sKSpace.lBaseResolution                  = '
			rx1 = re.compile(pattern1, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			# pattern2 = '<ParamLong."PhaseEncodingLines">  {*'
			# rx2 = re.compile(pattern2, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern3 = '<ParamString."ResonantNucleus">  { "*'
			rx3 = re.compile(pattern3, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			# pattern3 = 'sTXSPEC.asNucleusInfo\[0\].tNucleus        = '
			# rx3 = re.compile(pattern3, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern4 = 'sKSpace.lPartitions                      = '
			rx4 = re.compile(pattern4, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern5 = 'sProtConsistencyInfo.flNominalB0         = [0-9]+.[0-9]+'
			rx5 = re.compile(pattern5,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern6 = 'lAverages                                = [0-9]+'
			rx6 = re.compile(pattern6)
			
			pattern7= 'asCoilSelectMeas\[0\].asList\[[0-9]+\].sCoilElementID.tCoilID'
			rx7 = re.compile(pattern7,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern8= '<ParamDouble."flReadoutOSFactor">'
			rx8 = re.compile(pattern8,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern9= '<ParamLong."NLinMeas">  { '
			rx9 = re.compile(pattern9,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern10= '<ParamLong."NSlcMeas">  {'
			rx10 = re.compile(pattern10,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern11= 'sSliceArray.asSlice\[0\].dPhaseFOV         = [0-9]+.*[0-9]+'
			rx11 = re.compile(pattern11,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern12= 'sSliceArray.asSlice\[0\].dReadoutFOV       = '
			rx12 = re.compile(pattern12,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern13= 'sSliceArray.asSlice\[0\].dThickness        = '
			rx13 = re.compile(pattern13,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern14= 'alTR\[0\]                                  = '
			rx14 = re.compile(pattern14,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern15= 'alTE\[0\]                                  = '
			rx15 = re.compile(pattern15,re.IGNORECASE|re.MULTILINE|re.DOTALL)	
			
			pattern16= 'adFlipAngleDegree\[0\]                     = '
			rx16 = re.compile(pattern16,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern17= 'lTotalScanTimeSec                        = '
			rx17 = re.compile(pattern17,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern18= 'sWiPMemBlock.adFree\[12\]                  = '
			rx18 = re.compile(pattern18,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern19= 'sWiPMemBlock.adFree\[11\]                  = [0-9]+.[0-9]+'
			rx19 = re.compile(pattern19,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern20= 'sWiPMemBlock.alFree\[5\]                   = [0-9]+'
			rx20 = re.compile(pattern20,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern21= 'sSliceArray.asSlice\[0\].sNormal.'
			rx21 = re.compile(pattern21,re.IGNORECASE|re.MULTILINE|re.DOTALL)	
				
			for a in rx1.findall(str(line)):
				nbPoints=([int(s) for s in line.split() if s.isdigit()])
			# for a in rx2.findall(str(line)):	
				# nbLines=([int(s) for s in line.split() if s.isdigit()])
			for a in rx3.findall(str(line)):	
				Nucleus=([str(s) for s in line.split()])
				Nucleus=str(Nucleus[2])
				Nucleus=Nucleus[1:-1]
			for a in rx4.findall(str(line)):	
				nbPartitions=([int(s) for s in line.split() if s.isdigit()])
			for a in rx5.findall(str(line)):	
				MagneticField=([str(s) for s in line.split()])
				MagneticField=MagneticField[2]
			for a in rx6.findall(str(line)):	
				averages=([int(s) for s in line.split() if s.isdigit()])
			maxcoils=0
			for a in rx7.findall(str(line)):	
				splits = a.split(']')
				for split in splits:
					if split == "" or split.rfind('[') == -1: break
					stringVal = split[split.rfind('[')+1:]
					coils = int(stringVal)
				if (coils) > maxcoils:
					maxcoils=(coils)
			for a in rx8.findall(str(line)):	
				res=([str(s) for s in line.split()])
				if len(res)>1:
					oversamplingFactor=res[4]
					oversamplingFactor=oversamplingFactor[0]
			for a in rx9.findall(str(line)):
				nbLines=([int(s) for s in line.split() if s.isdigit()])	
			for a in rx20.findall(str(line)):
				nbLines_ICEOFF=([int(s) for s in line.split() if s.isdigit()])
				
			for a in rx10.findall(str(line)):
				nbSlices=([int(s) for s in line.split() if s.isdigit()])
			for a in rx11.findall(str(line)):
				# PhaseFOV=([int(s) for s in line.split() if s.isdigit()])
				PhaseFOV=([str(s) for s in line.split()])
				PhaseFOV=PhaseFOV[2]
				PhaseFOV=float(PhaseFOV)
			for a in rx12.findall(str(line)):
				ReadoutFOV=([int(s) for s in line.split() if s.isdigit()])		
			for a in rx13.findall(str(line)):
				SliceThickness=([int(s) for s in line.split() if s.isdigit()])
			for a in rx14.findall(str(line)):
				TR=([int(s) for s in line.split() if s.isdigit()])		
			for a in rx15.findall(str(line)):
				TE=([int(s) for s in line.split() if s.isdigit()])	
			for a in rx16.findall(str(line)):
				FlipAngle=([int(s) for s in line.split() if s.isdigit()])
			for a in rx17.findall(str(line)):
				ACQTime=([int(s) for s in line.split() if s.isdigit()])
			for a in rx18.findall(str(line)):
				ROGradmax=([str(s) for s in line.split()])
				ROGradmax=ROGradmax[2]
				ROGradmax=float(ROGradmax)
			for a in rx19.findall(str(line)):
				BandWidth=([str(s) for s in line.split()])
				print BandWidth
				BandWidth=float(BandWidth[2])
			for a in rx21.findall(str(line)):
				Orientation=([str(s) for s in line.split()])	
				splits = Orientation[0].split('.')
				Orientation=str(splits[3])
			if not nbLines:
				nbLines=nbLines_ICEOFF
				
			if source.tell() > int(header_size[0]):
				break
				
		if nbPoints: 				print 'INFO    : Number of points per line =  ',int(nbPoints[0])
		if nbLines : 				print 'INFO    : Number of Lines =  ',int(nbLines[0])
		# print 'INFO    : Number of Fourier Lines =  ',int(nbLines[0])
		if nbSlices: 				print 'INFO    : Number of Slices =  ',int(nbSlices[0])
		if averages: 				print 'INFO    : Number of Averages = ', int(averages[0])
		if coils: 					print 'INFO    : Number of Coils = ', int(coils)+1
		if nbPartitions: 			print 'INFO    : Number of Partitions =  ',int(nbPartitions[0])
		if oversamplingFactor: 		print 'INFO    : Oversampling Factor =  ',oversamplingFactor
		if Nucleus: 				print 'INFO    : Used Nucleus = ',str(Nucleus)
		if MagneticField: 			print 'INFO    : Nominal Magnetic Field Strength = ', MagneticField, 'T'
		if ReadoutFOV and PhaseFOV: print 'INFO    : FOV = ', int(ReadoutFOV[0]), ' mm x',PhaseFOV,' mm'
		if SliceThickness: 			print 'INFO    : Slice Thickness = ', int(SliceThickness[0]),' mm'
		if FlipAngle: 				print 'INFO    : Flip Angle = ', int(FlipAngle[0]),' deg'
		if ROGradmax: 				print 'INFO    : RO Grad Max value = ', ROGradmax, ' mT / m'
		if BandWidth: 				print 'INFO    : BandWidth per Pixel = ', BandWidth, 'Hz / Pixel'
		if TE and TR: 				print 'INFO    : Sequence Times : TR =', int(TR[0]),'and TE =',int(TE[0]),' usec'
		if ACQTime: 				print 'INFO    : Acquisition Duration = ', int(ACQTime[0])

		if (str(Orientation) == str('dTra')):
			Orientation=str('Transverse')
		if (Orientation == str('dCor')):
			Orientation=str('Coronal')
		if (Orientation == str('dSag')):
			Orientation=str('Sagital')
		print 'INFO    : Slice Orientation = ', Orientation
		
		# if BandWidth :
			# if (str(Nucleus)==str('31P')):
				# T1=5
				# T2=0.06
			# if (str(Nucleus)==str('1H')):
				# T1=4,4
				# T2=0,04
			# estimated_SNR=(((1-np.exp(-(int(TR[0])/1e6)/T1))*np.sin(int(FlipAngle[0])*np.pi/180))/(1-(np.cos(int(FlipAngle[0])*np.pi/180)*np.exp(-(int(TR[0])/1e6)/T1))))*(np.exp(-(int(TE[0])/1e6)/T2))*((((int(ReadoutFOV[0])*PhaseFOV)/(int(nbLines[0])*int(nbPoints[0])*int(oversamplingFactor)))*int(SliceThickness[0])*np.sqrt(int(nbPoints[0])*int(oversamplingFactor)*int(nbLines[0])*int(averages[0])))/(np.sqrt(BandWidth)))
			# print 'INFO    : EstimatedSNR = ',estimated_SNR
		
		# testAlexandre=True
		# if testAlexandre:
		
			# oversamplingFactor=1
			# DataLineLength=int(nbPoints[0])*2*int(oversamplingFactor)*32
			# position = source.seek(0,os.SEEK_END)
			# is_end = source.tell() == os.fstat(source.fileno()).st_size
			# print is_end
			# source.seek(-DataLineLength, os.SEEK_END)
			# Kline=source.read((DataLineLength/8+(128)))		# we read a line
			# hdr=Kline[0:127]
			# signal=Kline[128:]
			# print hdr
			# return
		slice=0	
		CplxDataFrame = np.zeros(shape=(int(nbSlices[0]),coils,int(nbLines[0]),int(nbPoints[0])*int(oversamplingFactor)), dtype=np.complex64)		
		if not HeaderOnly:
			# In bits we have N times nbpoints due to oversampling (oversampling is not present in the number of points)
			# We acquire complex data so : Imaginary and Real part (x2) each stored on 32bits (4octets)
			# There is a header block of 128 bits = 16octets for each acquired line
			# [ 128 bits flags ] [ DataLineLength bits of data ]
			DataLineLength=int(nbPoints[0])*2*int(oversamplingFactor)*32
			# print DataLineLength
			
			# Go to begining of Data
			position = source.seek(int(header_size[0]))
			is_end = source.tell() == os.fstat(source.fileno()).st_size
			
			p=0
			# While we don't reach EOF
			while (is_end == False):
				Kline=source.read((DataLineLength/8+(128)))		# we read a line
				hdr=Kline[0:127]
				signal=Kline[128:]
						
				# Processing K space line Header
				# ulFlagsAndDMALength=hdr[0:4]
				# lMeasUID = hdr[4:8]
				# ulScanCounter = hdr[8:12]
				# print struct.unpack('I', hdr[8:12])			# to get K space line ID
				# ulTimeStamp = hdr[12:16]
				# ulPMUTimeStamp = hdr[16:20]
				# aulEvalInfoMask= hdr[20:24]
				# aulEvalInfoMask= hdr[24:28]
				# ushSamplesinScans=hdr[28:30]
				# UsedChannel=hdr[30:32]				
				# sLoopCounter = hdr[32:60]
				# sCutOff = hdr[60:64]
				# ushKspaceCentreColumn= hdr[64:66]
				# ushCoilSelect = hdr[66:68] 
				# fReadOutOffCentre=hdr[68:72]
				# ulTimeSinceLastRF=hdr[72:76]
				# ushKspaceCentreLineNo=hdr[76:78]
				# ushKspaceCenterPartitionNo=hdr[78:80] 
				# aushICEProgramPara=hdr[80:88]
				# aushFreePara=hdr[88:96]
				# sSD=hdr[96:124]
				# ushChannelId = hdr[124:126]
				# print struct.unpack('h', hdr[124:126])				# To get used coil channel
				# ushPTABPosNeg = hdr[126:128]
				
				pos=0
				# if  int(struct.unpack('I', hdr[8:12])[0]) >= int(nbLines[0]):
					# slice+=1
				if (int(struct.unpack('h', hdr[124:126])[0]))!=0 and str(Nucleus) != str('1H'):			# If not Proton coil
					while (pos !=len(signal)) :
						# Read in Octets (Bytes), each value is coded over 32bits = 32/8=4octets
						realdouble=(struct.unpack('f', signal[pos:pos+4]))
						imdouble=(struct.unpack('f', signal[pos+4:pos+8]))
						cplx.append(complex(realdouble[0],imdouble[0]))
						# print slice
						# coil = int(struct.unpack('h', hdr[124:126])[0])-16
						# line = int(struct.unpack('I', hdr[8:12])[0])-1
						# point=  pos/8
						# CplxDataFrame[slice][coil][line][point]=CplxDataFrame[slice][coil][line][point]+complex(realdouble[0],imdouble[0])
						pos = pos+8
				
				if (str(Nucleus) == str('1H')):
					while (pos !=len(signal)):
						# Read in Octets (Bytes), each value is coded over 32bits = 32/8=4octets
						realdouble=(struct.unpack('f', signal[pos:pos+4]))
						imdouble=(struct.unpack('f', signal[pos+4:pos+8]))
						cplx.append(complex(realdouble[0],imdouble[0]))
						pos = pos+8
				
				is_end = source.tell() == os.fstat(source.fileno()).st_size
				
			if verbose: print ("INFO    : Instanciating Output Matrix")
			
			if (str(Nucleus) == str('1H')):
				coils=int(coils+1)
			CplxDataFrame = np.zeros(shape=(int(nbSlices[0]),int(averages[0]),coils,int(nbLines[0]),int(nbPoints[0])*int(oversamplingFactor)), dtype=np.complex64)	
			for slice in range(int(nbSlices[0])):
				for av in range(int(averages[0])):
					for y in range(int(nbLines[0])):
						for coil in range(coils):
							for x in range(int(nbPoints[0])*int(oversamplingFactor)):
								CplxDataFrame[slice][av][coil][y][x]=cplx[slice*int(averages[0])*int(nbLines[0])*(int(coils))*int(nbPoints[0])*int(oversamplingFactor)+av*int(nbLines[0])*(int(coils))*int(nbPoints[0])*int(oversamplingFactor)+y*(int(coils))*int(nbPoints[0])*int(oversamplingFactor)+coil*int(nbPoints[0])*int(oversamplingFactor)+x]
			source.close()
			
			# Storing Acquisition Parameters
			ACQParams=[]
			ACQParams.append(int(nbLines[0]))
			ACQParams.append(int(nbPoints[0]))
			ACQParams.append(int(averages[0]))
			ACQParams.append(coils)
			ACQParams.append(int(nbSlices[0]))
			ACQParams.append(int(oversamplingFactor))
			ACQParams.append(Nucleus)
			ACQParams.append(MagneticField)
			ACQParams.append(int(ReadoutFOV[0]))
			ACQParams.append((PhaseFOV))
			ACQParams.append(int(SliceThickness[0]))
			ACQParams.append(int(FlipAngle[0]))
			ACQParams.append(Orientation)
			ACQParams.append(int(TR[0]))
			ACQParams.append(int(TE[0]))
			if ROGradmax : ACQParams.append(float(ROGradmax))
			else : ACQParams.append(None)
			if BandWidth : ACQParams.append(float(BandWidth))
			else : ACQParams.append(None)
			# Visualization tools :
			if visualization:
				for slice in range(int(nbSlices[0])):
					# for coil in range(coils):
						# PlotImg(np.absolute(CplxDataFrame[slice][0][coil][:][:])+np.absolute(CplxDataFrame[slice][1][coil][:][:]))
						# import processingFunctions as pf
						# print 'Mean intensity of image = ',pf.MeanImage(np.absolute(CplxDataFrame[slice][0][coil][:][:])+np.absolute(CplxDataFrame[slice][1][coil][:][:]))
					PlotImg(np.absolute(CplxDataFrame[slice][0][0][:][:]))
		else:
			CplxDataFrame=[]
			ACQParams=[]
	#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
	print ("------------------------------------------------------------")
	return CplxDataFrame,ACQParams
	
def ExtractDataFromRawData_TPI(source_file,verbose,viz,HeaderOnly):
	# READ RAWDATA FROM MRI ACQUISITION ON SIEMENS MACHINE
	
	#from ctypes import windll,c_ulong
	#STD_OUTPUT_HANDLE_ID = c_ulong(0xfffffff5)
	#windll.Kernel32.GetStdHandle.restype = c_ulong
	#std_output_hdl = windll.Kernel32.GetStdHandle(STD_OUTPUT_HANDLE_ID)

	# color 10=vert 12=rouge
	#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
	print "------------------------------------------------------------"
	print "INFO    : Running SiRaw Version 2.0"		# SIemens Raw
	print
	
	if viz: 
		visualization=True
		print 'INFO    : Visualization is Enabled'
	else : 
		print 'INFO    : Vizualization is Disabled !'
		visualization=False

	cplx=[]
	trajectory=[]
	# An update in sequence code on January 15th, 2015 allows to return ROGradmax and Bandwith per pixel
	# Those fields don't exist in previous rawdata file so we initialize them to empty values
	ROGradmax=[];BandWidth=[];nbLines_ICEOFF=[];nbLines=[];hdr=[];signal=[]				
	with open(source_file, 'rb') as source:
		if verbose: print 'INFO    : Extracting acquisition parameters'
		header_size=source.read(4)
		header_size=struct.unpack('L', header_size)
		if verbose: print 'INFO    : Header size = ',int(header_size[0])
		
		for line in source: 
		
			pattern1 = '<ParamLong."BaseResolution">  {*'
			rx1 = re.compile(pattern1, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			# #pattern2 = '<ParamLong."PhaseEncodingLines">  {*'
			# #rx2 = re.compile(pattern2, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern3 = '<ParamString."ResonantNucleus">  { "*'
			rx3 = re.compile(pattern3, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern4 = '<ParamLong."Partitions">  { "*'
			rx4 = re.compile(pattern4, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern5 = 'sProtConsistencyInfo.flNominalB0         = [0-9]+.[0-9]+'
			rx5 = re.compile(pattern5,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern6 = 'lAverages                                = [0-9]+'
			rx6 = re.compile(pattern6)
			
			pattern7= 'asCoilSelectMeas\[0\].asList\[[0-9]+\].sCoilElementID.tCoilID'
			rx7 = re.compile(pattern7,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern8= '<ParamDouble."flReadoutOSFactor">'
			rx8 = re.compile(pattern8,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern9= '<ParamLong."NLinMeas">  { '
			rx9 = re.compile(pattern9,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern10= '<ParamLong."NSlcMeas">  {'
			rx10 = re.compile(pattern10,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern11= 'sSliceArray.asSlice\[0\].dPhaseFOV         = [0-9]+.*[0-9]+'
			rx11 = re.compile(pattern11,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern12= 'sSliceArray.asSlice\[0\].dReadoutFOV       = '
			rx12 = re.compile(pattern12,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern13= 'sSliceArray.asSlice\[0\].dThickness        = '
			rx13 = re.compile(pattern13,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern14= 'alTR\[0\]                                  = '
			rx14 = re.compile(pattern14,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern15= 'alTE\[0\]                                  = '
			rx15 = re.compile(pattern15,re.IGNORECASE|re.MULTILINE|re.DOTALL)	
			
			pattern16= 'adFlipAngleDegree\[0\]                     = '
			rx16 = re.compile(pattern16,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern17= 'lTotalScanTimeSec                        = '
			rx17 = re.compile(pattern17,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern18= 'sWiPMemBlock.adFree\[12\]                  = '
			rx18 = re.compile(pattern18,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern19= 'sWiPMemBlock.alFree\[11\]                  = [0-9]+.[0-9]+'
			rx19 = re.compile(pattern19,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern20= 'sWiPMemBlock.alFree\[5\]                   = [0-9]+'
			rx20 = re.compile(pattern20,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern21= 'sSliceArray.asSlice\[0\].sNormal.'
			rx21 = re.compile(pattern21,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			for a in rx1.findall(str(line)):
				nbPoints=([int(s) for s in line.split() if s.isdigit()])
			# #for a in rx2.findall(str(line)):	
				# #nbLines=([int(s) for s in line.split() if s.isdigit()])
			for a in rx3.findall(str(line)):	
				Nucleus=([str(s) for s in line.split()])
				Nucleus=str(Nucleus[2])
				Nucleus=Nucleus[1:-1]
			for a in rx4.findall(str(line)):	
				nbPartitions=([int(s) for s in line.split() if s.isdigit()])
			for a in rx5.findall(str(line)):	
				MagneticField=([str(s) for s in line.split()])
				MagneticField=MagneticField[2]
			for a in rx6.findall(str(line)):	
				averages=([int(s) for s in line.split() if s.isdigit()])
			maxcoils=0
			for a in rx7.findall(str(line)):	
				splits = a.split(']')
				for split in splits:
					if split == "" or split.rfind('[') == -1: break
					stringVal = split[split.rfind('[')+1:]
					coils = int(stringVal)
				if (coils) > maxcoils:
					maxcoils=(coils)
			for a in rx8.findall(str(line)):	
				res=([str(s) for s in line.split()])
				if len(res)>1:
					oversamplingFactor=res[4]
					oversamplingFactor=oversamplingFactor[0]
			for a in rx9.findall(str(line)):
				nbLines=([int(s) for s in line.split() if s.isdigit()])	
			for a in rx20.findall(str(line)):
				nbLines_ICEOFF=([int(s) for s in line.split() if s.isdigit()])
				
			for a in rx10.findall(str(line)):
				nbSlices=([int(s) for s in line.split() if s.isdigit()])
			for a in rx11.findall(str(line)):
				# #PhaseFOV=([int(s) for s in line.split() if s.isdigit()])
				PhaseFOV=([str(s) for s in line.split()])
				PhaseFOV=PhaseFOV[2]
				PhaseFOV=float(PhaseFOV)
			for a in rx12.findall(str(line)):
				ReadoutFOV=([int(s) for s in line.split() if s.isdigit()])		
			for a in rx13.findall(str(line)):
				SliceThickness=([int(s) for s in line.split() if s.isdigit()])
			for a in rx14.findall(str(line)):
				TR=([int(s) for s in line.split() if s.isdigit()])		
			for a in rx15.findall(str(line)):
				TE=([int(s) for s in line.split() if s.isdigit()])	
			for a in rx16.findall(str(line)):
				FlipAngle=([int(s) for s in line.split() if s.isdigit()])
			for a in rx17.findall(str(line)):
				ACQTime=([int(s) for s in line.split() if s.isdigit()])
			for a in rx18.findall(str(line)):
				ROGradmax=([str(s) for s in line.split()])
				ROGradmax=ROGradmax[2]
				ROGradmax=float(ROGradmax)
			for a in rx19.findall(str(line)):
				BandWidth=([int(s) for s in line.split() if s.isdigit()])
				BandWidth=int(BandWidth[0])
			for a in rx21.findall(str(line)):
				Orientation=([str(s) for s in line.split()])	
				splits = Orientation[0].split('.')
				Orientation=str(splits[3])
			if not nbLines:
				nbLines=nbLines_ICEOFF
				
			if source.tell() > int(header_size[0]):
				break
				
		print 'INFO    : Number of points per line =  ',int(nbPoints[0])
		print 'INFO    : Number of Lines =  ',int(nbLines[0])
		# print 'INFO    : Number of Fourier Lines =  ',int(nbLines[0])
		print 'INFO    : Number of Slices =  ',int(nbSlices[0])
		print 'INFO    : Number of Averages = ', int(averages[0])
		print 'INFO    : Number of Coils = ', int(coils)+1
		print 'INFO    : Number of Partitions =  ',int(nbPartitions[0])
		print 'INFO    : Oversampling Factor =  ',oversamplingFactor
		print 'INFO    : Used Nucleus = ',str(Nucleus)
		print 'INFO    : Nominal Magnetic Field Strength = ', MagneticField, 'T'
		print 'INFO    : FOV = ', int(ReadoutFOV[0]), ' mm x',PhaseFOV,' mm'
		print 'INFO    : Slice Thickness = ', int(SliceThickness[0]),' mm'
		print 'INFO    : Flip Angle = ', int(FlipAngle[0]),' deg'
		if ROGradmax: print 'INFO    : RO Grad Max value = ', ROGradmax, ' mT / m'
		if BandWidth: print 'INFO    : BandWidth per Pixel = ', BandWidth, 'Hz / Pixel'
		print 'INFO    : Sequence Times : TR =', int(TR[0]),'and TE =',int(TE[0]),' usec'
		print 'INFO    : Acquisition Duration = ', int(ACQTime[0])

		if (str(Orientation) == str('dTra')):
			Orientation=str('Transverse')
		if (Orientation == str('dCor')):
			Orientation=str('Coronal')
		if (Orientation == str('dSag')):
			Orientation=str('Sagital')
		print 'INFO    : Slice Orientation = ', Orientation
		
		if BandWidth :
			if (str(Nucleus)==str('31P')):
				T1=2
				T2=0.06
			if (str(Nucleus)==str('1H')):
				T1=4,4
				T2=0,04
			# estimated_SNR=(((1-np.exp(-(int(TR[0])/1e6)/T1))*np.sin(int(FlipAngle[0])*np.pi/180))/(1-(np.cos(int(FlipAngle[0])*np.pi/180)*np.exp(-(int(TR[0])/1e6)/T1))))*(np.exp(-(int(TE[0])/1e6)/T2))*((((int(ReadoutFOV[0])*PhaseFOV)/(int(nbLines[0])*int(nbPoints[0])*int(oversamplingFactor)))*int(SliceThickness[0])*np.sqrt(int(nbPoints[0])*int(oversamplingFactor)*int(nbLines[0])*int(averages[0])))/(np.sqrt(BandWidth)))
			# print 'INFO    : EstimatedSNR = ',estimated_SNR
			
		if not HeaderOnly:
			# In bits we have N times nbpoints due to oversampling (oversampling is not present in the number of points)
			# We acquire complex data so : Imaginary and Real part (x2) each stored on 32bits (4octets)
			# There is a header block of 128 bits = 16octets for each acquired line
			# [ 128 bits flags ] [ DataLineLength bits of data ]
			DataLineLength=int(nbPoints[0])*2*int(oversamplingFactor)*32
			# print DataLineLength
			
			# Go to begining of Data
			position = source.seek(int(header_size[0]))
			is_end = source.tell() == os.fstat(source.fileno()).st_size
			
			p=0;KX=[];KY=[];KZ=[]
			# While we don't reach EOF

			while (is_end == False and p<130):
				HDRlength=192			# Position parts have an extra hearder of 64 octets so 128 + 64 = 192
				PosLength=int(nbPoints[0])*int(oversamplingFactor)*4
				Kline=source.read((DataLineLength/8+(128))*(coils+1)+3*(PosLength+HDRlength))		# we read a block
				# print "projection ",p
				
				# The accurate header length is 60octets, but there is a 4octet ending flag (control ?) so we need to adjust position
				KSpace_hdr1=Kline[0:HDRlength-4]
				Kx=Kline[HDRlength-4:HDRlength+PosLength-4]
				KSpace_hdr2=Kline[HDRlength+PosLength-4:2*HDRlength+PosLength-4]
				Ky=Kline[2*HDRlength+PosLength-4:2*HDRlength+2*PosLength-4]
				KSpace_hdr3=Kline[2*HDRlength-4+2*PosLength:3*HDRlength+2*PosLength-4]
				Kz=Kline[3*HDRlength+2*PosLength-4:3*HDRlength+3*PosLength-4]
				
				for i in range(coils+1):	
					hdr=Kline[i*((DataLineLength/8)+128)+3*(HDRlength+PosLength):(i*(DataLineLength/8))+((i+1)*128)+3*(HDRlength+PosLength)]
					signal=Kline[(i*(DataLineLength/8))+((i+1)*128)+3*(HDRlength+PosLength):((i+1)*(DataLineLength/8))+((i+1)*128)+3*(HDRlength+PosLength)]
					# print struct.unpack('f',signal[0:4])

					# Processing K space line Header
					ulFlagsAndDMALength=hdr[0:4]
					lMeasUID = hdr[4:8]
					ulScanCounter = hdr[8:12]
					ulTimeStamp = hdr[12:16]
					ulPMUTimeStamp = hdr[16:20]
					aulEvalInfoMask= hdr[20:24]
					aulEvalInfoMask= hdr[24:28]
					ushSamplesinScans=hdr[28:30]
					UsedChannel=hdr[30:32]	
					sLoopCounter = hdr[32:60]
					sCutOff = hdr[60:64]
					ushKspaceCentreColumn= hdr[64:66]
					ushCoilSelect = hdr[66:68] 
					fReadOutOffCentre=hdr[68:72]
					ulTimeSinceLastRF=hdr[72:76]
					ushKspaceCentreLineNo=hdr[76:78]
					ushKspaceCenterPartitionNo=hdr[78:80] 
					aushICEProgramPara=hdr[80:88]
					aushFreePara=hdr[88:96]
					sSD=hdr[96:124]
					ushChannelId = hdr[124:126]
					ushPTABPosNeg = hdr[126:128]

					pos=0
					while (pos !=len(signal)):
						# Read in Octets (Bytes), each value is coded over 32bits = 32/8=4octets
						realdouble=(struct.unpack('f', signal[pos:pos+4]))
						imdouble=(struct.unpack('f', signal[pos+4:pos+8]))
						cplx.append(complex(realdouble[0],imdouble[0]))
						pos = pos+8
					
					pos2=0
					while (pos2 !=len(Kx)):
						#Read Positions and store them in vector
						KX.append(struct.unpack('f', Kx[pos2:pos2+4]))
						KY.append(struct.unpack('f', Ky[pos2:pos2+4]))
						KZ.append(struct.unpack('f', Kz[pos2:pos2+4]))
						pos2=pos2+4
				p=p+1
				is_end = source.tell() == os.fstat(source.fileno()).st_size
			
			# print len(KX)
			if verbose: print ("INFO    : Instanciating Output Matrix")
			# print len(cplx)
			CplxDataFrame2 =np.array(cplx)
			CplxDataFrame = np.zeros(shape=(int(averages[0]),coils+1,p+1,int(nbPoints[0])*int(oversamplingFactor)), dtype=np.complex64)	
			for av in range(int(averages[0])):
				for y in range(p):
					for coil in range(coils+1):
						for x in range(int(nbPoints[0])*int(oversamplingFactor)):
							CplxDataFrame[av][coil][y][x]=cplx[av*int(p)*(int(coils)+1)*int(nbPoints[0])*int(oversamplingFactor)+y*(int(coils)+1)*int(nbPoints[0])*int(oversamplingFactor)+coil*int(nbPoints[0])*int(oversamplingFactor)+x]
			source.close()
			KX = np.array(KX)
			KY = np.array(KY)
			KZ = np.array(KZ)
			print CplxDataFrame.shape
			# Storing Acquisition Parameters
			ACQParams=[]
			ACQParams.append(p+1)
			ACQParams.append(int(nbPoints[0]))
			ACQParams.append(int(averages[0]))
			ACQParams.append(coils+1)
			ACQParams.append(int(nbSlices[0]))
			ACQParams.append(int(oversamplingFactor))
			ACQParams.append(Nucleus)
			ACQParams.append(MagneticField)
			ACQParams.append(int(ReadoutFOV[0]))
			ACQParams.append((PhaseFOV))
			ACQParams.append(int(SliceThickness[0]))
			ACQParams.append(int(FlipAngle[0]))
			if ROGradmax : ACQParams.append(ROGradmax)
			if BandWidth : ACQParams.append(BandWidth)
			ACQParams.append(Orientation)
	
			# Visualization tools :
			if visualization:
				for slice in range(int(nbSlices[0])):
					PlotImg(np.absolute(CplxDataFrame[0][0][:][:]))
				
		else:
			CplxDataFrame=[]
			ACQParams=[]
	#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
	print ("------------------------------------------------------------")
	return CplxDataFrame2,ACQParams,KX,KY,KZ
	
	
def BrukerRawData(FIDsourcefile,METHODsourcefile,TRAJsourcefile,HeaderOnly):

	print "------------------------------------------------------------"
	print "INFO    : Running BrukerRawData Version 1.0"		
	print
	
	with open(METHODsourcefile,"r") as source:
		nbLines=[];Nucleus=[];Method=[];KspacePoints=[];nbAverages=[];TR=[]; nbSlices=[];TA=[];TE=[];ExcitationPulse=[];
		linefov=0;linereso=0;lineslice=0;lineTA=0;linecoefr=[];linecoefp=[];linecoefs=[];coefsr=[];coefsp=[];coefss=[];
		lineSliceOrientation=0; SliceThickness=[];SliceOrientation=[];SliceMode=[];SliceGap=[];lineSliceMode=0;lineSliceGap=0;
		for i, line in enumerate (source): 
			pattern2 = '\#\#\$Method='
			rx2 = re.compile(pattern2, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern1 = '\#\#\$NPro='
			rx1 = re.compile(pattern1, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern3 = '\#\#\$PVM_RepetitionTime='
			rx3 = re.compile(pattern3, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern4 = '\#\#\$PVM_NAverages='
			rx4 = re.compile(pattern4, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern5 = '\#\#\$PVM_Nucleus1Enum='
			rx5 = re.compile(pattern5, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern6 = '\#\#\$PVM_Fov='
			rx6 = re.compile(pattern6, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern7 = '\#\#\$PVM_SpatResol='
			rx7 = re.compile(pattern7, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern8 = '\#\#\$traj_r='
			rx8 = re.compile(pattern8, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern9 = '\#\#\$PVM_GradCalConst='
			rx9 = re.compile(pattern9, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern10 = '\#\#\$traj_r='
			rx10 = re.compile(pattern10, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern11 = '\#\#\$traj_p='
			rx11 = re.compile(pattern11, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern12 = '\#\#\$traj_s='
			rx12 = re.compile(pattern12, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern13 = '\#\#\$PVM_SPackArrNSlices='
			rx13 = re.compile(pattern13, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern14 = '\#\#\$PVM_ScanTimeStr='
			rx14 = re.compile(pattern14, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern15 = '\#\#\$PVM_EchoTime='
			rx15 = re.compile(pattern15, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern16 = '\#\#\$PVM_SPackArrSliceOrient'
			rx16 = re.compile(pattern16, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern17 = '\#\#\$PVM_SPackArrSliceGapMode='
			rx17 = re.compile(pattern17, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern18 = '\#\#\$PVM_SPackArrSliceGap='
			rx18 = re.compile(pattern18, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern19 = '\#\#\$ExcPulseEnum='
			rx19 = re.compile(pattern19, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern20 = '\#\#\$PVM_SliceThick='
			rx20 = re.compile(pattern20, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			for a in rx1.findall(str(line)):
				r=line.split('=')
				nbLines=r[1]
			for a in rx2.findall(str(line)):	
				Method=([str(s) for s in line.split()])
				splits = Method[0].split('=')
				for split in splits:
					if split == "" : break
					stringVal = split[split.rfind('=')+1:]
					Method=stringVal
			for a in rx3.findall(str(line)):
				r=line.split('=')
				TR=r[1]
			for a in rx4.findall(str(line)):
				r=line.split('=')
				nbAverages=r[1]
			for a in rx5.findall(str(line)):	
				Nucleus=([str(s) for s in line.split()])
				splits = Nucleus[0].split('=')
				for split in splits:
					if split == "" : break
					stringVal = split[split.rfind('=')+1:]
					Nucleus=stringVal
			for a in rx6.findall(str(line)):	
				FOV=([int(s) for s in line.split() if s.isdigit()])
				linefov = i
			for a in rx7.findall(str(line)):	
				Resolution=([int(s) for s in line.split() if s.isdigit()])
				linereso= i
			for a in rx8.findall(str(line)):	
				KspacePoints=([str(s) for s in line.split()])
				KspacePoints=KspacePoints[1]
			for a in rx9.findall(str(line)):	
				r=line.split('=')
				Gamma=r[1]
			for a in rx10.findall(str(line)):	
				linecoefr = int(i)
			for a in rx11.findall(str(line)):
				linecoefp = i	
			for a in rx12.findall(str(line)):	
				linecoefs = i	
				
			if linefov != 0 and i == linefov+1:
				ra=line.split('\n')
				r=ra[0].split(' ')
				FOVx=r[0]
				FOVy=r[1]
				if len(r) >2:
					FOVz=r[2]
				
			if linereso != 0 and i == linereso+1:
				ra=line.split('\n')
				r=ra[0].split(' ')
				RESx=r[0]
				RESy=r[1]
				if len(ra) >2:
					RESz=r[2]
				else:
					RESz=1
			
			
			if linecoefr != 0 and i > linecoefr and i < linecoefr+13:
					values=line.split(' ')
					for j in range(len(values)):
						if values[j] != '\n':
							coefsr.append(float(values[j]))
			
			if linecoefp != 0 and i > linecoefp and i < linecoefp+13:
					values=line.split(' ')
					for j in range(len(values)):
						if values[j] != '\n':
							coefsp.append(float(values[j]))
							
			if linecoefs != 0 and i > linecoefs and i < linecoefs+13:
					values=line.split(' ')
					for j in range(len(values)):
						if values[j] != '\n':
							coefss.append(float(values[j]))	

			for a in rx13.findall(str(line)):	
				Slices=([int(s) for s in line.split() if s.isdigit()])
				lineslice= i
				
			if lineslice != 0 and i == lineslice+1:
				rb=line.split('\n')
				nbSlices=rb[0]
				
			for a in rx14.findall(str(line)):	
				tta=([int(s) for s in line.split() if s.isdigit()])
				lineTA= i
				
			if lineTA != 0 and i == lineTA+1:
				rc=line.split('\n')
				TA=rc[0]
				
			for a in rx15.findall(str(line)):
				r=line.split('=')
				TE=r[1].split('\n')
				TE=TE[0]
				
			for a in rx16.findall(str(line)):	
				ts=([int(s) for s in line.split() if s.isdigit()])
				lineSliceOrientation= i
				
			if lineSliceOrientation != 0 and i == lineSliceOrientation+1:
				rc=line.split('\n')
				SliceOrientation=rc[0]	
				
			for a in rx17.findall(str(line)):	
				ts=([int(s) for s in line.split() if s.isdigit()])
				lineSliceMode= i
				
			if lineSliceMode != 0 and i == lineSliceMode+1:
				rc=line.split('\n')
				SliceMode=rc[0]	

			for a in rx18.findall(str(line)):	
				so=([int(s) for s in line.split() if s.isdigit()])
				lineSliceGap= i
				
			if lineSliceGap != 0 and i == lineSliceGap+1:
				rc=line.split('\n')
				SliceGap=rc[0]	
				
			for a in rx19.findall(str(line)):
				r=line.split('=')
				ExcitationPulse=r[1]
				
			for a in rx20.findall(str(line)):
				r=line.split('=')
				SliceThickness=r[1].split('\n')[0]
			
		source.close()
		print 'INFO    : BRUKER MRI DATA '
		if Method : 			print 'INFO    : Sequence = ',str(Method)
		if Nucleus :			print 'INFO    : Used Nucleus = ',str(Nucleus)
		if KspacePoints : 		print 'INFO    : Points per line = ',int(KspacePoints)
		if nbLines:				print 'INFO    : Number of Lines = ',int(nbLines)
		if nbSlices:			print 'INFO    : Number of Slices = ',int(nbSlices)
		if nbAverages:			print 'INFO    : Nb of Acquired Averages = ',int(nbAverages)
		if TR: 					print 'INFO    : TR = ',int(TR), ' ms'
		if TE: 					print 'INFO    : TE = ',float(TE), ' ms'
		if TA:					print 'INFO    : Acquisition Time = ',str(TA)
		if len(r) >2: 			print 'INFO    : FOV = ',float(FOVx), 'mm x',float(FOVy), 'mm x', float(FOVz), 'mm'
		else : 		  			print 'INFO    : FOV = ',float(FOVx), 'mm x',float(FOVy),'mm'
		if len(ra) >2:  		print 'INFO    : Resolution = ',float(RESx), 'mm x',float(RESy), 'mm x', float(RESz), 'mm'
		else :					print 'INFO    : Resolution = ',float(RESx), 'mm x',float(RESy), 'mm '
		if SliceThickness:		print 'INFO    : Slice Thickness =',str(SliceThickness), 'mm'
		if Gamma:				print 'INFO    : Gyromagnetic Ratio =',float(Gamma), 'kHz'
		if SliceOrientation:	print 'INFO    : Slice Orientation =',str(SliceOrientation)
		if SliceMode:			print 'INFO    : Slice Mode =',str(SliceMode)
		if SliceGap:			print 'INFO    : Slice Gap =',float(SliceGap)
		if ExcitationPulse:		print 'INFO    : Excitation Pulse =',str(ExcitationPulse)
		
		SizeFIDfile=os.path.getsize(FIDsourcefile)
		# print (SizeFIDfile)
		EffectivePoints=int(SizeFIDfile/(int(nbLines)*4*2))
		if EffectivePoints != KspacePoints :
			ZeroFilling=True
			ZFnb = EffectivePoints-int(KspacePoints)
		else :
			ZeroFilling=False
		
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 12)
		if ZeroFilling : print 'WARNING : ZERO FILLING !'
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
		if ZeroFilling : print 'INFO    : Number of Zeros Padded per line=',int(ZFnb)

	# /!\ THERE WILL ALWAYS BE ONLY ONE AVERAGE AS BRUKER MRI OS IS SUMMING AVERAGES !
	nbAverages=1

	if not HeaderOnly:	
		with open(TRAJsourcefile, "r") as pos:
			trajpos= np.fromfile(pos, dtype=np.float64)	
			# print(len(trajpos))
			Kx=trajpos[0::3]
			Ky=trajpos[1::3]
			Kz=trajpos[2::3]
			pos.close()
	
		with open(FIDsourcefile, "r") as f:
			data= np.fromfile(f, dtype=np.int32)			# Long integer in Matlab ==> 32bits / 4 octets
			cplxdata = data[0::2] + 1j * data[1::2]
			# print (data[0::2])
			# print (data[1::2])
		
		# print len(cplxdata)
		rawKspaceMatrix=np.zeros(shape=(int(nbLines),int(KspacePoints)))
		rawKspaceMatrix=np.reshape(cplxdata,(int(nbLines),int(EffectivePoints)))
		
		print "INFO    : Matrix size = ",rawKspaceMatrix.shape
		
		ACQParams=[]
		ACQParams.append(int(nbLines))
		ACQParams.append(int(KspacePoints))
		ACQParams.append(int(nbAverages))
		ACQParams.append(str(Nucleus))
		ACQParams.append(float(FOVx))
		ACQParams.append(float(FOVy))
		ACQParams.append(float(FOVz))
		ACQParams.append(float(RESx))
		ACQParams.append(float(RESy))
		ACQParams.append(float(RESz))
		ACQParams.append(int(EffectivePoints))
		# ACQParams.append(coils)
		# ACQParams.append(int(nbSlices[0]))
		# ACQParams.append(MagneticField)
		print '------------------------------------------------------------'
		return rawKspaceMatrix,Kx,Ky,Kz,ACQParams,coefsr,coefsp,coefss

# BrukerRawData('C://Users/AC243636/Documents/DataBrucker/13/fid','C://Users/AC243636/Documents/DataBrucker/13/method','C://Users/AC243636/Documents/DataBrucker/13/traj',False)

def ParseHeader(source_file):

# READ RAWDATA FROM MRI ACQUISITION ON SIEMENS MACHINE
	
	# color 10=vert 12=rouge
	#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
	print "------------------------------------------------------------"
	print "INFO    : Running SiRawParser Version 1.4"		# SIemens Raw
	print
	
	# An update in sequence code on January 15th, 2015 allows to return ROGradmax and Bandwith per pixel
	# Those fields don't exist in previous rawdata file so we initialize them to empty values
	ROGradmax=[];BandWidth=[];nbLines_ICEOFF=[];nbLines=[];nbPoints=[];nbSlices=[];averages=[];coils=[];nbPartitions=[];Orientation=[];
	oversamplingFactor=[];Nucleus=[];MagneticField=[];SliceThickness=[];ReadoutFOV=[];PhaseFOV=[];FlipAngle=[];TE=[];TR=[];ACQTime=[];
	NbProjections=[];TPI_linear_proportion=[];TPI_reso=[]; NbEchoes=[]; SpectreLength=[];SpectralPulseDuration=[]; LarmorHz=[]; Multiple_Echoes=[];maxechoes=0; OffsetPPM=[];
	with open(source_file, 'rb') as source:
		print 'INFO    : Extracting acquisition parameters'
		header_size=source.read(4)
		header_size=struct.unpack('I', header_size)
		print 'INFO    : Header size = ',int(header_size[0])
		
		for line in source: 
		
			# pattern1 = '<ParamLong."BaseResolution">  {*'
			# rx1 = re.compile(pattern1, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern1 = 'sKSpace.lBaseResolution                  = '
			rx1 = re.compile(pattern1, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			# pattern2 = '<ParamLong."PhaseEncodingLines">  {*'
			# rx2 = re.compile(pattern2, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern3 = '<ParamString."ResonantNucleus">  { "*'
			rx3 = re.compile(pattern3, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			# pattern3 = 'sTXSPEC.asNucleusInfo\[0\].tNucleus        = '
			# rx3 = re.compile(pattern3, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern4 = 'sKSpace.lPartitions                      = '
			rx4 = re.compile(pattern4, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern5 = 'sProtConsistencyInfo.flNominalB0         = [0-9]+.[0-9]+'
			rx5 = re.compile(pattern5,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern6 = 'lAverages                                = [0-9]+'
			rx6 = re.compile(pattern6)
			
			pattern7= 'asCoilSelectMeas\[0\].asList\[[0-9]+\].sCoilElementID.tCoilID'
			rx7 = re.compile(pattern7,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern8= '<ParamDouble."flReadoutOSFactor">'
			rx8 = re.compile(pattern8,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern9= '<ParamLong."NLinMeas">  { '
			rx9 = re.compile(pattern9,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern10= '<ParamLong."NSlcMeas">  {'
			rx10 = re.compile(pattern10,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern11= 'sSliceArray.asSlice\[0\].dPhaseFOV         = [0-9]+.*[0-9]+'
			rx11 = re.compile(pattern11,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern12= 'sSliceArray.asSlice\[0\].dReadoutFOV       = '
			rx12 = re.compile(pattern12,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern13= 'sSliceArray.asSlice\[0\].dThickness        = *'
			rx13 = re.compile(pattern13,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern14= 'alTR\[0\]                                  = '
			rx14 = re.compile(pattern14,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern15= 'alTE\[0\]                                  = '
			rx15 = re.compile(pattern15,re.IGNORECASE|re.MULTILINE|re.DOTALL)	
			
			pattern16= 'adFlipAngleDegree\[0\]                     = '
			rx16 = re.compile(pattern16,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern17= 'lTotalScanTimeSec                        = '
			rx17 = re.compile(pattern17,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern18= 'sWiPMemBlock.adFree\[12\]                  = '
			rx18 = re.compile(pattern18,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern19= 'sWiPMemBlock.adFree\[11\]                  = [0-9]+.[0-9]+'
			rx19 = re.compile(pattern19,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern20= 'sWiPMemBlock.alFree\[5\]                   = [0-9]+'
			rx20 = re.compile(pattern20,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern21= 'sSliceArray.asSlice\[0\].sNormal.'
			rx21 = re.compile(pattern21,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern22= 'sWiPMemBlock.alFree\[0\]                   = '
			rx22 = re.compile(pattern22,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern23= 'sWiPMemBlock.adFree\[1\]                   = *'
			rx23 = re.compile(pattern23,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern24= 'sWiPMemBlock.adFree\[0\]                   = '
			rx24 = re.compile(pattern24,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern25= 'sSpecPara.lVectorSize                    = *'
			rx25 = re.compile(pattern25, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern26 = 'sTXSPEC.asNucleusInfo\[0\].lFrequency      = *'
			rx26 = re.compile(pattern26, re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern27= 'alTE\[[0-9]+\]                                  = [0-9]+'
			rx27 = re.compile(pattern27,re.IGNORECASE|re.MULTILINE|re.DOTALL)	
			
			pattern28= 'lContrasts                               = [0-9]+'
			rx28 = re.compile(pattern28,re.IGNORECASE|re.MULTILINE|re.DOTALL)

			pattern29= 'sWiPMemBlock.adFree\[2\]                   = '
			rx29 = re.compile(pattern29,re.IGNORECASE|re.MULTILINE|re.DOTALL)
			
			pattern30= 'sWiPMemBlock.alFree\[3\]                   = '
			rx30 = re.compile(pattern30,re.IGNORECASE|re.MULTILINE|re.DOTALL)

						
			for a in rx1.findall(str(line)):
				nbPoints=([int(s) for s in line.split() if s.isdigit()])
			# for a in rx2.findall(str(line)):	
				# nbLines=([int(s) for s in line.split() if s.isdigit()])
			for a in rx3.findall(str(line)):	
				Nucleus=([str(s) for s in line.split()])
				Nucleus=str(Nucleus[2])
				Nucleus=Nucleus[1:-1]
			for a in rx4.findall(str(line)):	
				nbPartitions=([int(s) for s in line.split() if s.isdigit()])
			for a in rx5.findall(str(line)):	
				MagneticField=([str(s) for s in line.split()])
				MagneticField=MagneticField[2]
			for a in rx6.findall(str(line)):	
				averages=([int(s) for s in line.split() if s.isdigit()])
			maxcoils=0
			for a in rx7.findall(str(line)):	
				splits = a.split(']')
				for split in splits:
					if split == "" or split.rfind('[') == -1: break
					stringVal = split[split.rfind('[')+1:]
					coils = int(stringVal)
				if (coils) > maxcoils:
					maxcoils=(coils)
			for a in rx8.findall(str(line)):	
				res=([str(s) for s in line.split()])
				if len(res)>1:
					oversamplingFactor=res[4]
					oversamplingFactor=oversamplingFactor[0]
			for a in rx9.findall(str(line)):
				nbLines=([int(s) for s in line.split() if s.isdigit()])	
			for a in rx20.findall(str(line)):
				nbLines_ICEOFF=([int(s) for s in line.split() if s.isdigit()])
				
			for a in rx10.findall(str(line)):
				nbSlices=([int(s) for s in line.split() if s.isdigit()])
			for a in rx11.findall(str(line)):
				# PhaseFOV=([int(s) for s in line.split() if s.isdigit()])
				PhaseFOV=([str(s) for s in line.split()])
				PhaseFOV=PhaseFOV[2]
				PhaseFOV=float(PhaseFOV)
			for a in rx12.findall(str(line)):
				ReadoutFOV=([int(s) for s in line.split() if s.isdigit()])		
			for a in rx13.findall(str(line)):
				SliceThickness=([str(s) for s in line.split()])
				SliceThickness=str(SliceThickness[2])
			for a in rx14.findall(str(line)):
				TR=([int(s) for s in line.split() if s.isdigit()])		
			for a in rx15.findall(str(line)):
				TE=([int(s) for s in line.split() if s.isdigit()])	
			for a in rx16.findall(str(line)):
				FlipAngle=([int(s) for s in line.split() if s.isdigit()])
			for a in rx17.findall(str(line)):
				ACQTime=([int(s) for s in line.split() if s.isdigit()])
			for a in rx18.findall(str(line)):
				ROGradmax=([str(s) for s in line.split()])
				ROGradmax=ROGradmax[2]
				ROGradmax=float(ROGradmax)
			for a in rx19.findall(str(line)):
				BandWidth=([str(s) for s in line.split()])
				print BandWidth
				BandWidth=float(BandWidth[2])
			for a in rx21.findall(str(line)):
				Orientation=([str(s) for s in line.split()])	
				splits = Orientation[0].split('.')
				Orientation=str(splits[3])
				
			for a in rx22.findall(str(line)):
				NbProjections=([int(s) for s in line.split() if s.isdigit()])
				
			for a in rx23.findall(str(line)):
				TPI_linear_proportion=([str(s) for s in line.split()])
				TPI_linear_proportion=float(TPI_linear_proportion[2])
				
			for a in rx24.findall(str(line)):
				TPI_reso=line.split()
				# TPI_reso=([int(s) for s in line.split() if s.isdigit()])
				TPI_reso=float(TPI_reso[2])
				
			for a in rx25.findall(str(line)):
				SpectreLength=([int(s) for s in line.split() if s.isdigit()])	
				SpectreLength= int(SpectreLength[0])
				
			for a in rx26.findall(str(line)):
				LarmorHz=([int(s) for s in line.split() if s.isdigit()])
				LarmorHz = int(LarmorHz[0])
			
			for a in rx27.findall(str(line)):
				# print(a.split())
				splited = a.split()
				Multiple_Echoes.append(splited[2])
			
				echoes = splited[0].split('[')
				echoes = echoes[1].split(']')

				if (int(echoes[0]) > maxechoes):
					maxechoes=int(echoes[0])
				Multiple_Echoes=Multiple_Echoes[0:maxechoes+1]
				
			for a in rx28.findall(str(line)):
				NbEchoes=([int(s) for s in line.split() if s.isdigit()])
				NbEchoes = int(NbEchoes[0])
				# print NbEchoes
				
			for a in rx29.findall(str(line)):
				OffsetPPM=line.split('=')
				OffsetPPM = float(OffsetPPM[1])	
				
			for a in rx30.findall(str(line)):
				SpectralPulseDuration=line.split('=')
				SpectralPulseDuration = float(SpectralPulseDuration[1])		
				
			if not nbLines:
				nbLines=nbLines_ICEOFF
				
			if source.tell() > int(header_size[0]):
				break
		source.close()		
			
		if nbPoints: 					print 'INFO    : Number of points per line =  ',int(nbPoints[0])
		if nbLines : 					print 'INFO    : Number of Lines =  ',int(nbLines[0])
		if NbProjections :				print 'INFO    : Number of Projections = ',int(NbProjections[0])
		# print 'INFO    : Number of Fourier Lines =  ',int(nbLines[0])
		if nbSlices: 					print 'INFO    : Number of Slices =  ',int(nbSlices[0])
		if averages: 					print 'INFO    : Number of Averages = ', int(averages[0])
		if coils : 						print 'INFO    : Number of Coils = ', int(coils)+1
		if coils ==0 : 					print 'INFO    : Number of Coils = ', int(coils)+1
		if nbPartitions: 				print 'INFO    : Number of Partitions =  ',int(nbPartitions[0])
		if oversamplingFactor: 			print 'INFO    : Oversampling Factor =  ',oversamplingFactor
		if Nucleus: 					print 'INFO    : Used Nucleus = ',str(Nucleus)
		if MagneticField: 				print 'INFO    : Nominal Magnetic Field Strength = ', MagneticField, 'T'
		if ReadoutFOV and PhaseFOV: 	print 'INFO    : FOV = ', int(ReadoutFOV[0]), ' mm x',PhaseFOV,' mm'
		if SliceThickness: 				print 'INFO    : Slice Thickness = ', str(SliceThickness),' mm'
		if FlipAngle: 					print 'INFO    : Flip Angle = ', int(FlipAngle[0]),' deg'
		if ROGradmax: 					print 'INFO    : RO Grad Max value = ', ROGradmax, ' mT / m'
		if BandWidth: 					print 'INFO    : BandWidth per Pixel = ', BandWidth, 'Hz / Pixel'
		if int(maxechoes) == 0:
			if TE and TR: 				print 'INFO    : Sequence Times : TR =', int(TR[0]),'and TE =',int(TE[0])
		elif int(maxechoes) > 1 and int(NbEchoes) != 1:
										maxechoes = NbEchoes
										print 'INFO    : Sequence Times : TR =', int(TR[0]), 'us'
										print 'INFO    : Multiple Echo Acquisition with',int(maxechoes), 'echoes'
										print 'INFO    : Echo Times : ', ', '.join(Multiple_Echoes[0:maxechoes]), 'us'
										

		elif int(maxechoes) > 1 and NbEchoes == 1:
										print 'INFO    : Single Echo Acquisition'
										maxechoes=1
		if ACQTime: 					print 'INFO    : Acquisition Duration = ', int(ACQTime[0]),' sec'
		if TPI_linear_proportion :  	print 'INFO    : TPI linear ratio = ', float(TPI_linear_proportion)
		if TPI_reso :  					print 'INFO    : TPI Spatial Resolution = ', float(TPI_reso),' mm'
		if SpectreLength :				print 'INFO    : Spectrum Size = ', int(SpectreLength)
		if LarmorHz :					print 'INFO    : Larmor frequency = ', int(LarmorHz), 'Hz'
		if OffsetPPM :					print 'INFO    : Excitation frequency offset = ', int(OffsetPPM), 'ppm'
		if SpectralPulseDuration :		print 'INFO    : Spectral Pulse Duration = ', int(SpectralPulseDuration), 'ms'
		
		if (str(Orientation) == str('dTra')):
			Orientation=str('Transverse')
		if (Orientation == str('dCor')):
			Orientation=str('Coronal')
		if (Orientation == str('dSag')):
			Orientation=str('Sagital')
		print 'INFO    : Slice Orientation = ', Orientation
		
		ACQParams=[]
		ACQParams.append(int(nbLines[0]))
		ACQParams.append(int(nbPoints[0]))
		ACQParams.append(int(averages[0]))
		ACQParams.append(coils)
		ACQParams.append(int(nbSlices[0]))
		ACQParams.append(int(oversamplingFactor))
		ACQParams.append(str(Nucleus))
		ACQParams.append(MagneticField)
		ACQParams.append(int(ReadoutFOV[0]))
		ACQParams.append((PhaseFOV))
		ACQParams.append(str(SliceThickness))
		ACQParams.append(int(FlipAngle[0]))
		ACQParams.append(Orientation)
		ACQParams.append(int(TR[0]))
		ACQParams.append(int(TE[0]))
		if ROGradmax : ACQParams.append(float(ROGradmax))
		else : ACQParams.append(None)
		if BandWidth : ACQParams.append(float(BandWidth))
		else : ACQParams.append(None)
		ACQParams.append(int(header_size[0]))
		if NbProjections : ACQParams.append(int(NbProjections[0]))
		else : ACQParams.append(None)
		ACQParams.append(int(nbPartitions[0]))
		if TPI_linear_proportion : ACQParams.append(float(TPI_linear_proportion))
		else : ACQParams.append(None)
		if TPI_reso : ACQParams.append(float(TPI_reso))
		else : ACQParams.append(None)
		if SpectreLength : ACQParams.append(int(SpectreLength))
		else : ACQParams.append(None)
		if LarmorHz : ACQParams.append(int(LarmorHz))
		else : ACQParams.append(None)
		ACQParams.append(int(maxechoes))
		# ACQParams.append(3)
		if Multiple_Echoes : ACQParams.append(Multiple_Echoes)
		else  : ACQParams.append(None)
		
	return ACQParams

def GetData(source_file,verbose,ACQParams):

	header_size = int(ACQParams[17])
	nbLines = int(ACQParams[0])
	nbPoints = int(ACQParams[1])
	averages = int(ACQParams[2])
	coils = int(ACQParams[3])
	oversamplingFactor = int(ACQParams[5])
	nbSlices = int(ACQParams[4])
	
	ligne=0
	
	with open(source_file, 'rb') as source:
		if verbose: print 'INFO    : Extracting acquisition parameters'
		DataLineLength=(nbPoints)*2*int(oversamplingFactor)*32
		# print DataLineLength
		
		# Go to begining of Data
		position = source.seek(header_size)
		is_end = source.tell() == os.fstat(source.fileno()).st_size
		

		# While we don't reach EOF
		while (is_end == False):
			Kline=source.read((DataLineLength/8+(128)))		# we read a line
			hdr=Kline[0:127]
			signal=Kline[128:]
			
			if ligne >= nbLines :
				ligne=0
		
			pos=0
			
			if (int(struct.unpack('h', hdr[124:126])[0]))!=0 and str(Nucleus) != str('1H'):			# If not Proton coil
				while (pos !=len(signal)):
					# Read in Octets (Bytes), each value is coded over 32bits = 32/8=4octets
					realdouble=(struct.unpack('f', signal[pos:pos+4]))
					imdouble=(struct.unpack('f', signal[pos+4:pos+8]))
					cplx.append(complex(realdouble[0],imdouble[0]))
					pos = pos+8
			
			if (str(Nucleus) == str('1H')):
				while (pos !=len(signal)):
					# Read in Octets (Bytes), each value is coded over 32bits = 32/8=4octets
					realdouble=(struct.unpack('f', signal[pos:pos+4]))
					imdouble=(struct.unpack('f', signal[pos+4:pos+8]))
					cplx.append(complex(realdouble[0],imdouble[0]))
					pos = pos+8
			
			is_end = source.tell() == os.fstat(source.fileno()).st_size
			
		if verbose: print ("INFO    : Instanciating Output Matrix")
		
		if (str(Nucleus) == str('1H')):
			coils=int(coils+1)
		CplxDataFrame = np.zeros(shape=(int(nbSlices[0]),int(averages[0]),coils,int(nbLines[0]),int(nbPoints[0])*int(oversamplingFactor)), dtype=np.complex64)	
		for slice in range(int(nbSlices[0])):
			for av in range(int(averages[0])):
				for y in range(int(nbLines[0])):
					for coil in range(coils):
						for x in range(int(nbPoints[0])*int(oversamplingFactor)):
							CplxDataFrame[slice][av][coil][y][x]=cplx[slice*int(averages[0])*int(nbLines[0])*(int(coils))*int(nbPoints[0])*int(oversamplingFactor)+av*int(nbLines[0])*(int(coils))*int(nbPoints[0])*int(oversamplingFactor)+y*(int(coils))*int(nbPoints[0])*int(oversamplingFactor)+coil*int(nbPoints[0])*int(oversamplingFactor)+x]
		source.close()
		ligne += 1
		
	return 	CplxDataFrame
	
def ReadKline (source_file, position, nbPoints, oversamplingFactor, NbCoils,Nucleus):
	with open(source_file, 'rb') as source:
		cplx=[]
		source.seek(position) # We go to desired position in file
		KspaceLine = np.zeros(shape=(int(nbPoints)*int(oversamplingFactor)),dtype=np.complex64)
		# A line in K space is acquired by each coil and composed of (IM,RE)*NbPts*Oversampling*32bits
		DataLineLength=(nbPoints)*2*int(oversamplingFactor)*32
		Kline=source.read((DataLineLength/8+(128)))		# we read a line (for a single coil)
		hdr=Kline[0:127]
		signal=Kline[128:]
		dec=False
		pos=0
		if (int(struct.unpack('h', hdr[124:126])[0]))!=0 and str(Nucleus) != str('1H'):			# If not Proton coil
			keep=True
			while (pos !=len(signal)):
				# Read in Octets (Bytes), each value is coded over 32bits = 32/8=4octets
				realdouble=(struct.unpack('f', signal[pos:pos+4]))
				imdouble=(struct.unpack('f', signal[pos+4:pos+8]))
				cplx.append(complex(realdouble[0],imdouble[0]))
				# np.append(KspaceLine,complex(realdouble[0],imdouble[0]))
				pos = pos+8

		if (str(Nucleus) == str('1H')):
			keep=True
			while (pos !=len(signal)):
				# Read in Octets (Bytes), each value is coded over 32bits = 32/8=4octets
				realdouble=(struct.unpack('f', signal[pos:pos+4]))
				imdouble=(struct.unpack('f', signal[pos+4:pos+8]))
				cplx.append(complex(realdouble[0],imdouble[0]))
				# np.append(KspaceLine,complex(realdouble[0],imdouble[0]))
				pos = pos+8
		
		if (int(struct.unpack('h', hdr[124:126])[0]))==0 and str(Nucleus) != str('1H'):			# Remove Proton coil in Non proton Acquisition
			keep=False
		
		if keep :
			for x in range(int(nbPoints)*int(oversamplingFactor)):
				KspaceLine[x] = cplx[x]
		
	source.close()
	position=position+len(Kline)
	return KspaceLine, position
	
def ReadFastSiemensRAD(Source,HeaderOnly):

	from ReadRawData import ParseHeader, ReadKline
	ACQParams = ParseHeader(Source)
	header_size = int(ACQParams[17])
	nbLines = int(ACQParams[0])
	nbPoints = int(ACQParams[1])
	averages = int(ACQParams[2])
	coils = int(ACQParams[3])
	oversamplingFactor = int(ACQParams[5])
	nbSlices = int(ACQParams[4])
	Nucleus = str(ACQParams[6])
	position=header_size
	
	# averages = 2
	coils=int(coils+1)
	CplxDataFrame = np.zeros(shape=(int(nbSlices),int(averages),coils,int(nbLines),int(nbPoints)*int(oversamplingFactor)), dtype=np.complex64)
	if not HeaderOnly:
		# CplxDataFrame = np.zeros(shape=(int(nbSlices),int(averages),coils,int(nbLines),int(nbPoints)*int(oversamplingFactor)), dtype=np.complex64)
		for slice in range(int(nbSlices)):
			for av in range(int(averages)):
				for line in range (nbLines):
					for coil in range(coils) :
						Cplxline, position = ReadKline(Source, position, nbPoints, oversamplingFactor, coils, Nucleus)
						CplxDataFrame[slice,av,coil,line]=Cplxline
						
		if (str(Nucleus) != str('1H')):	
			CplxDataFrame=np.delete(CplxDataFrame,0,2)
	return	CplxDataFrame, ACQParams
	
def ReadTPIBlock(source_file, position, nbPoints, oversamplingFactor,coils):
	with open(source_file, 'rb') as source:
		cplx=[]
		source.seek(position) # We go to desired position in file
		KspaceLine = np.zeros(shape=(int(coils)*int(nbPoints)*int(oversamplingFactor)),dtype=np.complex64)
		# KXvec = np.zeros(shape=(int(nbPoints)),dtype=np.float32)
		# KYvec = np.zeros(shape=(int(nbPoints)),dtype=np.float32)
		# KZvec = np.zeros(shape=(int(nbPoints)),dtype=np.float32)
		# A line in K space is acquired by each coil and composed of (IM,RE)*NbPts*Oversampling*32bits
		# Data are NOT supposed to be Oversampled in TPI
		
		KX=[];KY=[];KZ=[]
		# While we don't reach EOF
		DataLineLength=int(nbPoints)*2*int(oversamplingFactor)*32

		HDRlength=192			# Position parts have an extra hearder of 64 octets so 128 + 64 = 192
		# HDRlength=188			# Position parts have an extra hearder of 64 octets so 128 + 64 = 192
		PosLength=int(nbPoints)*int(oversamplingFactor)*4
		Kline=source.read((DataLineLength/8+(128))*(coils)+3*(PosLength+HDRlength))		# we read a block
		
		# The accurate header length is 60octets, but there is a 4octet ending flag (control ?) so we need to adjust position
		
		# the old one still to be checked
		# KSpace_hdr1=Kline[0:HDRlength-4]
		# Kx=Kline[HDRlength-4:HDRlength+PosLength-4]
		# KSpace_hdr2=Kline[HDRlength+PosLength-4:2*HDRlength+PosLength-4]
		# Ky=Kline[2*HDRlength+PosLength-4:2*HDRlength+2*PosLength-4]
		# KSpace_hdr3=Kline[2*HDRlength-4+2*PosLength:3*HDRlength+2*PosLength-4]
		# Kz=Kline[3*HDRlength+2*PosLength-4:3*HDRlength+3*PosLength-4]
		
		KSpace_hdr1=Kline[0:HDRlength-4]
		Kx=Kline[HDRlength-4:HDRlength+PosLength-4]
		KSpace_hdr2=Kline[HDRlength+PosLength-4:2*HDRlength+PosLength-8]
		Ky=Kline[2*HDRlength+PosLength-8:2*HDRlength+2*PosLength-8]
		KSpace_hdr3=Kline[2*HDRlength-8+2*PosLength:3*HDRlength+2*PosLength-12]
		Kz=Kline[3*HDRlength+2*PosLength-12:3*HDRlength+3*PosLength-12]
		
		for i in range(coils):	
			hdr=Kline[i*((DataLineLength/8)+128)+3*(HDRlength+PosLength):(i*(DataLineLength/8))+((i+1)*128)+3*(HDRlength+PosLength)]
			signal=Kline[(i*(DataLineLength/8))+((i+1)*128)+3*(HDRlength+PosLength):((i+1)*(DataLineLength/8))+((i+1)*128)+3*(HDRlength+PosLength)]
			# print struct.unpack('f',signal[0:4])

			pos=0
			while (pos !=len(signal)):
				# Read in Octets (Bytes), each value is coded over 32bits = 32/8=4octets
				realdouble=(struct.unpack('f', signal[pos:pos+4]))
				imdouble=(struct.unpack('f', signal[pos+4:pos+8]))
				cplx.append(complex(realdouble[0],imdouble[0]))
				pos = pos+8
			
			pos2=0
			while (pos2 !=len(Kx)):
				#Read Positions and store them in vector
				KX.append(struct.unpack('f', Kx[pos2:pos2+4]))
				KY.append(struct.unpack('f', Ky[pos2:pos2+4]))
				KZ.append(struct.unpack('f', Kz[pos2:pos2+4]))
				pos2=pos2+4
		
	source.close()
	# print (KX,KY,KZ)
	# print len(cplx)
	for x in range(int(coils)*int(nbPoints)*int(oversamplingFactor)):
		KspaceLine[x] = cplx[x]
	position=position+len(Kline)
	# del(KX);del(KY); del(KZ); 
	del(cplx); del(pos2); del(pos)
	# del(KSpace_hdr1); del(Kx); del(KSpace_hdr2); del(Ky); del(KSpace_hdr3); del(Kz)

	return KspaceLine,KX, KY, KZ, position
	
def Read_NS_TPIBlock(source_file, position, nbPoints, oversamplingFactor,coils):
	with open(source_file, 'rb') as source:
		cplx=[]
		source.seek(position) # We go to desired position in file
		KspaceLine = np.zeros(shape=(int(coils)*int(nbPoints)*int(oversamplingFactor)),dtype=np.complex64)
		# A line in K space is acquired by each coil and composed of (IM,RE)*NbPts*Oversampling*32bits
		# Data are NOT supposed to be Oversampled in TPI
		
		KX=[];KY=[];KZ=[]
		# While we don't reach EOF
		DataLineLength=int(nbPoints)*2*int(oversamplingFactor)*32

		HDRlength=192			# Position parts have an extra hearder of 64 octets so 128 + 64 = 192
		# HDRlength=188			# Position parts have an extra hearder of 64 octets so 128 + 64 = 192
		PosLength=int(nbPoints)*int(oversamplingFactor)*4
		Kline=source.read((DataLineLength/8+(128))*(coils)+3*(PosLength+HDRlength))		# we read a block
		
		# The accurate header length is 60octets, but there is a 4octet ending flag (control ?) so we need to adjust position
		
		# the old one still to be checked
		# KSpace_hdr1=Kline[0:HDRlength-4]
		# Kx=Kline[HDRlength-4:HDRlength+PosLength-4]
		# KSpace_hdr2=Kline[HDRlength+PosLength-4:2*HDRlength+PosLength-4]
		# Ky=Kline[2*HDRlength+PosLength-4:2*HDRlength+2*PosLength-4]
		# KSpace_hdr3=Kline[2*HDRlength-4+2*PosLength:3*HDRlength+2*PosLength-4]
		# Kz=Kline[3*HDRlength+2*PosLength-4:3*HDRlength+3*PosLength-4]
		
		KSpace_hdr1=Kline[(coils)*(DataLineLength/8)+((coils)*128):(coils)*(DataLineLength/8)+((coils)*128)+HDRlength-4]
		Kx=Kline[(coils)*(DataLineLength/8)+((coils)*128)+HDRlength-4:(coils)*(DataLineLength/8)+((coils)*128)+HDRlength+PosLength-4]
		KSpace_hdr2=Kline[(coils)*(DataLineLength/8)+((coils)*128)+HDRlength+PosLength-4:(coils)*(DataLineLength/8)+((coils)*128)+2*HDRlength+PosLength-8]
		Ky=Kline[(coils)*(DataLineLength/8)+((coils)*128)+2*HDRlength+PosLength-8:(coils)*(DataLineLength/8)+((coils)*128)+2*HDRlength+2*PosLength-8]
		KSpace_hdr3=Kline[(coils)*(DataLineLength/8)+((coils)*128)+2*HDRlength-8+2*PosLength:(coils)*(DataLineLength/8)+((coils)*128)+3*HDRlength+2*PosLength-12]
		Kz=Kline[(coils)*(DataLineLength/8)+((coils)*128)+3*HDRlength+2*PosLength-12:(coils)*(DataLineLength/8)+((coils)*128)+3*HDRlength+3*PosLength-12]
		
		for i in range(coils):	
			hdr=Kline[i*((DataLineLength/8)+128):(i*(DataLineLength/8))+((i+1)*128)]
			signal=Kline[(i*(DataLineLength/8))+((i+1)*128):((i+1)*(DataLineLength/8))+((i+1)*128)]
			# print struct.unpack('f',signal[0:4])

			pos=0
			while (pos !=len(signal)):
				# Read in Octets (Bytes), each value is coded over 32bits = 32/8=4octets
				realdouble=(struct.unpack('f', signal[pos:pos+4]))
				imdouble=(struct.unpack('f', signal[pos+4:pos+8]))
				cplx.append(complex(realdouble[0],imdouble[0]))
				# print complex(realdouble[0],imdouble[0])
				pos = pos+8
			
			pos2=0
			# print len(Kx),len(Ky),len(Kz)
			while (pos2 !=len(Kx)):
				#Read Positions and store them in vector
				KX.append(struct.unpack('f', Kx[pos2:pos2+4]))
				KY.append(struct.unpack('f', Ky[pos2:pos2+4]))
				KZ.append(struct.unpack('f', Kz[pos2:pos2+4]))
				pos2=pos2+4
		
		
	source.close()
	# print (KX,KY,KZ)
	# print len(cplx)
	for x in range(int(coils)*int(nbPoints)*int(oversamplingFactor)):
		KspaceLine[x] = cplx[x]
	position=position+len(Kline)
	# del(KX);del(KY); del(KZ); 
	del(cplx); del(pos2); del(pos)
	# del(KSpace_hdr1); del(Kx); del(KSpace_hdr2); del(Ky); del(KSpace_hdr3); del(Kz)

	return KspaceLine,KX, KY, KZ, position	
	
def ReadFastSiemensTPI(Source,HeaderOnly,NS_TPI):

	from ReadRawData import ParseHeader, ReadKline
	ACQParams = ParseHeader(Source)
	header_size = int(ACQParams[17])
	nbLines = int(ACQParams[0])
	nbPoints = int(ACQParams[1])
	averages = int(ACQParams[2])
	coils = int(ACQParams[3])
	oversamplingFactor = int(ACQParams[5])
	nbSlices = int(ACQParams[4])
	Nucleus = str(ACQParams[6])
	position=header_size
	# projections=8497  # Sandro data set
	# projections=19900
	if NS_TPI : 
		nbLines = ACQParams[18]
	else:
		nbLines=5000
	
	coils=int(coils+1)
	CplxDataFrame = np.zeros(shape=(averages,coils,nbLines,nbPoints*int(oversamplingFactor)), dtype=np.complex64)
	KXarray = np.zeros(shape=(nbLines,nbPoints*int(oversamplingFactor)), dtype=np.float32)	
	KYarray = np.zeros(shape=(nbLines,nbPoints*int(oversamplingFactor)), dtype=np.float32)	
	KZarray = np.zeros(shape=(nbLines,nbPoints*int(oversamplingFactor)), dtype=np.float32)	
	for av in range(int(averages)):
		for y in range(nbLines):
			if NS_TPI==False:
				Cplxline, KX, KY, KZ, position = ReadTPIBlock(Source, position, nbPoints, oversamplingFactor,coils)
			else:
				Cplxline, KX, KY, KZ, position = Read_NS_TPIBlock(Source, position, nbPoints, oversamplingFactor,coils)
			# print 'Reading Projection ', y
			for coil in range(coils):
				CplxDataFrame[av][coil][y]=Cplxline[coil*nbPoints*int(oversamplingFactor):(coil+1)*nbPoints*int(oversamplingFactor)]
			KXarray[y,:] = np.array(KX[0:nbPoints*int(oversamplingFactor)]).T
			KYarray[y,:] = np.array(KY[0:nbPoints*int(oversamplingFactor)]).T
			KZarray[y,:] = np.array(KZ[0:nbPoints*int(oversamplingFactor)]).T
			
	# if (str(Nucleus) != str('1H')):				# The case of the proton channel is handled in reconstruction
		# CplxDataFrame=np.delete(CplxDataFrame,0,1)
	
	print KXarray.shape,KYarray.shape, KZarray.shape, CplxDataFrame.shape
	return	CplxDataFrame, ACQParams, KXarray, KYarray, KZarray
	
# def ReadFastSiemens(Source):

	# from ReadRawData import ParseHeader, ReadKline
	# ACQParams = ParseHeader(Source)
	# header_size = int(ACQParams[17])
	# nbLines = int(ACQParams[0])
	# nbPoints = int(ACQParams[1])
	# averages = int(ACQParams[2])
	# coils = int(ACQParams[3])
	# oversamplingFactor = int(ACQParams[5])
	# nbSlices = int(ACQParams[4])
	# Nucleus = str(ACQParams[6])
	# position=header_size
	# if (str(Nucleus) == str('1H')):
				# coils=int(coils+1)
	# CplxDataFrame = np.zeros(shape=(int(nbSlices),int(averages),coils+1,int(nbLines),int(nbPoints)*int(oversamplingFactor)), dtype=np.complex64)
	# for slice in range(int(nbSlices)):
		# for av in range(int(averages)):
			# for line in range (nbLines):
				# for coil in range(coils) :
					# Cplxline, position = ReadKline(Source, position, nbPoints, oversamplingFactor, coils, Nucleus)
					# CplxDataFrame[slice,av,coil,line]=Cplxline
	# return	CplxDataFrame	
	
def ReadSiemensSpectro(source_file,verbose):

	from ReadRawData import ParseHeader
	ACQParams = ParseHeader(source_file)
	header_size = int(ACQParams[17])
	nbPoints = int(ACQParams[22])
	averages = int(ACQParams[2])
	coils = int(ACQParams[3])
	Nucleus = str(ACQParams[6])
	LarmorHz = int(ACQParams[23])
	B0 = float(ACQParams[7])
	position=header_size
	
	name = str(source_file).split('\\')
	filename = name[len(name)-1]
	sequencestr=str(filename).split('_')
	sequence=sequencestr[len(sequencestr)-3]
	print 'INFO    : Spectroscopy sequence = ', sequence
	ResonanceFreq = float(B0*LarmorHz)/(2*3.1415926535)
	print 'INFO    : Resonance Frequency = ', ResonanceFreq/1000000, 'MHz'
	print Nucleus
	if sequence == 'fid' and str(Nucleus)!='7Li':
		coils = coils+2
	if sequence == 'se' or str(Nucleus)=='7Li':
		coils = coils
	print coils
	# with open(source_file, 'rb') as source:
		# cplx=[]
		# source.seek(position) # We go to desired position in file
		# Spectrum = np.zeros(shape=(averages*(coils)*int(nbPoints)),dtype=np.complex64)
		
		# HDRlength=128
		# PosLength=int(nbPoints)*8
		# Spectrum=source.read(averages*(coils)*(PosLength+HDRlength))	
		# print len(Spectrum)
		# for j in range(averages):
			# for i in range(coils):	
				# hdr=Spectrum[i*((int(PosLength))+HDRlength):((i*int(PosLength))+((i+1)*HDRlength))]
				# signal=Spectrum[((i*(PosLength))+((i+1)*HDRlength)):(((i+1)*(PosLength))+((i+1)*HDRlength))]
				# pos=0
				# while (pos !=len(signal)):
					# ###Read in Octets (Bytes), each value is coded over 32bits = 32/8=4octets
					# realdouble=(struct.unpack('f', signal[pos:pos+4]))
					# imdouble=(struct.unpack('f', signal[pos+4:pos+8]))
					# cplx.append(complex(realdouble[0],imdouble[0]))
					# pos = pos+8
	# source.close()
	
	Spectrum = np.zeros(shape=(int(averages),int(coils)*int(nbPoints)),dtype=np.complex64)
	print Spectrum.shape
	blocksize = (coils)*(int(nbPoints)*8+128)
	for j in range(averages):
		position, cplx = ReadSpectroBlock(source_file, position, blocksize, nbPoints,coils)

		for x in range(int(coils)*int(nbPoints)):
			Spectrum[j,x] = cplx[x]	
	
	Spectrum = np.sum(Spectrum,0)
	Spectrum = np.reshape(Spectrum,(coils,nbPoints))
	
	print Spectrum.shape
	# position=position+len(Spectrum)
	# print nbPoints
	# print len(cplx)
	
	# for x in range(int(averages)*int(coils)*int(nbPoints)):
		# Spectrum[x] = cplx[x]	
	# print len(Spectrum)
	
	# Spectrum = np.reshape(Spectrum,(averages,coils,nbPoints))
	# Spectrum = np.sum(Spectrum,0)
	# print Spectrum.shape
	from visualization import PlotSpectrum
	if sequence == 'fid'  and str(Nucleus)!='7Li':
		PlotSpectrum(Spectrum[2,:]) # FID
		from matplotlib.mlab import find
		S=Spectrum[2,:]
		indices = find((np.real(S[1:]) >= 0) & (np.real(S[:-1]) < 0))
		crossings = [i - S[i] / (S[i+1] - S[i]) for i in indices]
		print 'Estimated frequency shift = ',float (np.real((ResonanceFreq / np.mean(np.diff(crossings)))))/1000000, 'ppm / ',float (np.real(ResonanceFreq / np.mean(np.diff(crossings))))/1000000*(ResonanceFreq/1000000), 'Hz'
	if sequence == 'se' or str(Nucleus)=='7Li':
		PlotSpectrum(Spectrum[0,:]) # SVS SE
		from matplotlib.mlab import find
		S=Spectrum[0,:]
		indices = find((np.real(S[1:]) >= 0) & (np.real(S[:-1]) < 0))
		crossings = [i - S[i] / (S[i+1] - S[i]) for i in indices]
		print 'Estimated frequency shift = ',float (np.real((ResonanceFreq / np.mean(np.diff(crossings)))))/1000000, 'ppm / ',float (np.real(ResonanceFreq / np.mean(np.diff(crossings))))/1000000*(ResonanceFreq/1000000), 'Hz'
	# PlotSpectrum(Spectrum)
	
	return Spectrum, position	

def ReadSpectroBlock(source_file,position,blocksize, nbPoints,coils):
	with open(source_file, 'rb') as source:
		cplx=[]
		source.seek(position) # We go to desired position in file
		# Spectrum = np.zeros(shape=(averages*(coils)*int(nbPoints)),dtype=np.complex64)
		
		HDRlength=128
		PosLength=int(nbPoints)*8
		Spectrum=source.read(blocksize)	
		for i in range(coils):	
			hdr=Spectrum[i*((int(PosLength))+HDRlength):((i*int(PosLength))+((i+1)*HDRlength))]
			signal=Spectrum[((i*(PosLength))+((i+1)*HDRlength)):(((i+1)*(PosLength))+((i+1)*HDRlength))]
			pos=0
			while (pos !=len(signal)):
				# Read in Octets (Bytes), each value is coded over 32bits = 32/8=4octets
				realdouble=(struct.unpack('f', signal[pos:pos+4]))
				imdouble=(struct.unpack('f', signal[pos+4:pos+8]))
				cplx.append(complex(realdouble[0],imdouble[0]))
				pos = pos+8
	source.close()
	
	position=position+blocksize
	return position, cplx

def ReadFastSiemensTPI_MultiEcho(Source,HeaderOnly,NS_TPI):

	from ReadRawData import ParseHeader, ReadKline
	ACQParams = ParseHeader(Source)
	header_size = int(ACQParams[17])
	nbLines = int(ACQParams[0])
	nbPoints = int(ACQParams[1])
	averages = int(ACQParams[2])
	coils = int(ACQParams[3])
	oversamplingFactor = int(ACQParams[5])
	nbSlices = int(ACQParams[4])
	Nucleus = str(ACQParams[6])
	nbEchoes=int(ACQParams[24])			# We return the max echoes +1 to get correct values

	position=header_size

	if NS_TPI : 
		nbLines = ACQParams[18]
	else:
		nbLines=5000
	
	coils=int(coils+1)
	CplxDataFrame = np.zeros(shape=(1,coils,nbLines,nbEchoes,nbPoints*int(oversamplingFactor)), dtype=np.complex64)
	KXarray = np.zeros(shape=(nbLines,nbPoints*int(oversamplingFactor)), dtype=np.float32)	
	KYarray = np.zeros(shape=(nbLines,nbPoints*int(oversamplingFactor)), dtype=np.float32)	
	KZarray = np.zeros(shape=(nbLines,nbPoints*int(oversamplingFactor)), dtype=np.float32)	
	for av in range(int(averages)):
		for y in range(nbLines):
			if NS_TPI==False:
				Cplxline, KX, KY, KZ, position = ReadTPIBlock(Source, position, nbPoints, oversamplingFactor,coils)
			else:
				Cplxline, KX, KY, KZ, position = Read_NS_ME_TPIBlock(Source, position, nbPoints, oversamplingFactor,coils,nbEchoes)
			# print 'Reading Projection ', y
			for echo in range(int(nbEchoes)):
				for coil in range(coils):
					CplxDataFrame[0][coil][y][echo]=CplxDataFrame[0][coil][y][echo]+Cplxline[(echo*coils*nbPoints*int(oversamplingFactor))+coil*nbPoints*int(oversamplingFactor):(echo*coils*nbPoints*int(oversamplingFactor))+(coil+1)*nbPoints*int(oversamplingFactor)]
			KXarray[y,:] = np.array(KX[0:nbPoints*int(oversamplingFactor)]).T
			KYarray[y,:] = np.array(KY[0:nbPoints*int(oversamplingFactor)]).T
			KZarray[y,:] = np.array(KZ[0:nbPoints*int(oversamplingFactor)]).T
			
	# if (str(Nucleus) != str('1H')):				# The case of the proton channel is handled in reconstruction
		# CplxDataFrame=np.delete(CplxDataFrame,0,1)
	
	print KXarray.shape,KYarray.shape, KZarray.shape, CplxDataFrame.shape
	return	CplxDataFrame, ACQParams, KXarray, KYarray, KZarray	
	
def Read_NS_ME_TPIBlock(source_file, position, nbPoints, oversamplingFactor,coils,echoes):
	with open(source_file, 'rb') as source:
		cplx=[]

		source.seek(position) # We go to desired position in file
		KspaceLine = np.zeros(shape=(int(echoes)*int(coils)*int(nbPoints)*int(oversamplingFactor)),dtype=np.complex64)
		# A line in K space is acquired by each coil and composed of (IM,RE)*NbPts*Oversampling*32bits
		# Data are NOT supposed to be Oversampled in TPI
		# In Multiple echoes mode data for each projection are stored as : [Coil1-echo1;Coil2-echo1;...;Coiln-echon;KX;KY;KZ]
		
		KX=[];KY=[];KZ=[]
		# Acquired Signal Data are still stored as N complex points stored over 32bits values for both real and Imaginary parts. --> N*2*32 *(possible Oversampling Factor) 
		DataLineLength=int(nbPoints)*2*int(oversamplingFactor)*32 # --> Length computed in Bits --> Later converted to Bytes (octets)

		HDRlength=192			# Position parts have an extra hearder of 64 octets so 128 + 64 = 192

		PosLength=int(nbPoints)*int(oversamplingFactor)*4			# Positions consist Nbpoints real value stored over 32bits --> 4 octets (possibly oversampled)
		# Kline=source.read((DataLineLength/8+(128))*(coils)+3*(PosLength+HDRlength))		# The old block was long of Nbcoils x Points + positions (3 times Positions (KX,KY,KZ))
		Kline=source.read((DataLineLength/8+(128))*(coils)*(echoes)+3*(PosLength+HDRlength))		# The new block is long of NbEchoes x NbCoils x Points + positions
		
		# The accurate header length is 60octets, but there is a 4octet ending flag (control ?) so we need to adjust position
		
		# the old one still to be checked
		# KSpace_hdr1=Kline[0:HDRlength-4]
		# Kx=Kline[HDRlength-4:HDRlength+PosLength-4]
		# KSpace_hdr2=Kline[HDRlength+PosLength-4:2*HDRlength+PosLength-4]
		# Ky=Kline[2*HDRlength+PosLength-4:2*HDRlength+2*PosLength-4]
		# KSpace_hdr3=Kline[2*HDRlength-4+2*PosLength:3*HDRlength+2*PosLength-4]
		# Kz=Kline[3*HDRlength+2*PosLength-4:3*HDRlength+3*PosLength-4]
		
		KSpace_hdr1=Kline[(echoes)*(coils)*(DataLineLength/8)+((echoes)*(coils)*128):(echoes)*(coils)*(DataLineLength/8)+((echoes)*(coils)*128)+HDRlength-4]
		Kx=Kline[(echoes)*(coils)*(DataLineLength/8)+((echoes)*(coils)*128)+HDRlength-4:(echoes)*(coils)*(DataLineLength/8)+((echoes)*(coils)*128)+HDRlength+PosLength-4]
		KSpace_hdr2=Kline[(echoes)*(coils)*(DataLineLength/8)+((echoes)*(coils)*128)+HDRlength+PosLength-4:(echoes)*(coils)*(DataLineLength/8)+((echoes)*(coils)*128)+2*HDRlength+PosLength-8]
		Ky=Kline[(echoes)*(coils)*(DataLineLength/8)+((echoes)*(coils)*128)+2*HDRlength+PosLength-8:(echoes)*(coils)*(DataLineLength/8)+((echoes)*(coils)*128)+2*HDRlength+2*PosLength-8]
		KSpace_hdr3=Kline[(echoes)*(coils)*(DataLineLength/8)+((echoes)*(coils)*128)+2*HDRlength-8+2*PosLength:(echoes)*(coils)*(DataLineLength/8)+((echoes)*(coils)*128)+3*HDRlength+2*PosLength-12]
		Kz=Kline[(echoes)*(coils)*(DataLineLength/8)+((echoes)*(coils)*128)+3*HDRlength+2*PosLength-12:(echoes)*(coils)*(DataLineLength/8)+((echoes)*(coils)*128)+3*HDRlength+3*PosLength-12]
		
		# print echoes, coils
		for e in range(echoes):
			for i in range(coils):	
				# print e,i
				hdr=Kline[e*(((coils)*(DataLineLength/8))+((coils)*128))+i*((DataLineLength/8)+128):e*(((coils)*(DataLineLength/8))+((coils)*128))+(i*(DataLineLength/8))+((i+1)*128)]
				signal=Kline[e*(((coils)*(DataLineLength/8))+((coils)*128))+(i*(DataLineLength/8))+((i+1)*128):e*(((coils)*(DataLineLength/8))+((coils)*128))+((i+1)*(DataLineLength/8))+((i+1)*128)]
				# print struct.unpack('f',signal[0:4])
				# print len(hdr), len(signal)
				pos=0
				while (pos !=len(signal)):
					# Read in Octets (Bytes), each value is coded over 32bits = 32/8=4octets
					realdouble=(struct.unpack('f', signal[pos:pos+4]))
					imdouble=(struct.unpack('f', signal[pos+4:pos+8]))
					cplx.append(complex(realdouble[0],imdouble[0]))
					# print complex(realdouble[0],imdouble[0])
					pos = pos+8
		pos2=0
		# print len(Kx),len(Ky),len(Kz)
		while (pos2 !=len(Kx)):
			#Read Positions and store them in vector
			KX.append(struct.unpack('f', Kx[pos2:pos2+4]))
			KY.append(struct.unpack('f', Ky[pos2:pos2+4]))
			KZ.append(struct.unpack('f', Kz[pos2:pos2+4]))
			pos2=pos2+4
		
		
	source.close()
	# print len(KX),len(KY),len(KZ)
	# print len(cplx)

	for x in range(int(echoes)*int(coils)*int(nbPoints)*int(oversamplingFactor)):
		KspaceLine[x] = cplx[x]
	position=position+len(Kline)
	# del(KX);del(KY); del(KZ); 
	del(cplx); del(pos2); del(pos); del(KSpace_hdr1); del(KSpace_hdr2); del(KSpace_hdr3);

	return KspaceLine,KX, KY, KZ, position		
	
	
def ReadFastSiemensTPI_MIA_MonoEcho(Source,HeaderOnly,NS_TPI):

	# In this case, we have 3 acquisitions at different excitation frequencies interleaved 
	# Therefore for a given projection, we acquire at 3 different frequency so 3 times (Single Echo here)

	from ReadRawData import ParseHeader, ReadKline
	ACQParams = ParseHeader(Source)
	header_size = int(ACQParams[17])
	nbLines = int(ACQParams[0])
	
	nbPoints = int(ACQParams[1])
	averages = int(ACQParams[2])
	coils = int(ACQParams[3])
	oversamplingFactor = int(ACQParams[5])
	nbSlices = int(ACQParams[4])
	Nucleus = str(ACQParams[6])
	nbEchoes=int(ACQParams[24])			# We return the max echoes +1 to get correct values

	nbLines=3*nbLines					# 3 times more projections to read but a given acqusition still have nbLines projections
	# nbEchoes=3
	
	position=header_size

	if NS_TPI : 
		nbLines = ACQParams[18]
	else:
		nbLines=5000
	
	coils=int(coils+1)
	
	# I have decided to use the echo dimension here to store the 3 different acquisition
	
	CplxDataFrame = np.zeros(shape=(1,coils,nbLines,3,nbPoints*int(oversamplingFactor)), dtype=np.complex64)
	KXarray = np.zeros(shape=(nbLines,nbPoints*int(oversamplingFactor)), dtype=np.float32)	
	KYarray = np.zeros(shape=(nbLines,nbPoints*int(oversamplingFactor)), dtype=np.float32)	
	KZarray = np.zeros(shape=(nbLines,nbPoints*int(oversamplingFactor)), dtype=np.float32)	
	for av in range(int(averages)):
		for y in range(nbLines):
			if NS_TPI==False:
				Cplxline, KX, KY, KZ, position = ReadTPIBlock(Source, position, nbPoints, oversamplingFactor,coils)
			else:
				Cplxline, KX, KY, KZ, position = Read_NS_ME_TPIBlock(Source, position, nbPoints, oversamplingFactor,coils,nbEchoes)
			# print 'Reading Projection ', 
			# print Cplxline.shape
			# print CplxDataFrame.shape
			for echo in range(int(nbEchoes)):
				for coil in range(coils):
					# print CplxDataFrame[0][coil][y][echo].shape
					CplxDataFrame[0][coil][round(y/3)][y%3]=CplxDataFrame[0][coil][round(y/3)][y%3]+Cplxline[(echo*coils*nbPoints*int(oversamplingFactor))+coil*nbPoints*int(oversamplingFactor):(echo*coils*nbPoints*int(oversamplingFactor))+(coil+1)*nbPoints*int(oversamplingFactor)]
			KXarray[y,:] = np.array(KX[0:nbPoints*int(oversamplingFactor)]).T
			KYarray[y,:] = np.array(KY[0:nbPoints*int(oversamplingFactor)]).T
			KZarray[y,:] = np.array(KZ[0:nbPoints*int(oversamplingFactor)]).T
			
	# if (str(Nucleus) != str('1H')):				# The case of the proton channel is handled in reconstruction
		# CplxDataFrame=np.delete(CplxDataFrame,0,1)
	
	print KXarray.shape,KYarray.shape, KZarray.shape, CplxDataFrame.shape
	return	CplxDataFrame, ACQParams, KXarray, KYarray, KZarray		
