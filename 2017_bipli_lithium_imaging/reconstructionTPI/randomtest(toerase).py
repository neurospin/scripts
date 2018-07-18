#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 14:41:23 2018

@author: js247994
"""
import io
import numpy as np

source_file="/neurospin/ciclops/projects/BIPLi7/ClinicalData/Raw_Data/2018_06_08/twix7T/meas_MID34_7Li_TPI_fisp_TR200_21deg_P05_5echos_FID2524.dat"
#source_file="/volatile/hithere.txt";
#with open("hithere.txt") as source:
position=524544
nbPoints=352
oversamplingFactor=1
coils=2
echoes=5

with open(source_file,'rb') as source:
    #source.seek(5)
    print("hello")
    #source2=source.decode("utf-8")
    #line.decode("utf-8")
    a="dljslfj"
    source.seek(0)
    print("\n")
    Kline=(source.read(12))
    print(Kline)
    #Kline2=Kline.encode('ISO-8850-1')
    cplx=[]
    #source=source.read().decode("UTF-8")
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
    int((DataLineLength/8+(128))*(coils)+3*(PosLength+HDRlength))
    Kline=source.read(int((DataLineLength/8+(128))*(coils)*(echoes)+3*(PosLength+HDRlength)))	# The old block was long of Nbcoils x Points + positions (3 times Positions (KX,KY,KZ))
    
    #Kline=source.read((DataLineLength/8+(128))*(coils)*(echoes)+3*(PosLength+HDRlength))		# The new block is long of NbEchoes x NbCoils x Points + positions

    for line in source:
        pattern1 = 'Hi'
        #rx1 = re.compile(pattern1, re.IGNORECASE|re.MULTILINE|re.DOTALL)    