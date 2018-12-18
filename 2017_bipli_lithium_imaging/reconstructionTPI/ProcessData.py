# -*- coding:Utf-8 -*-
#
# Author : Arthur Coste
# Date : December 2014
# Purpose : Process Siemens Raw Data File 
#---------------------------------------------------------------------------------------------------------------------------------
# command : ProcessData.py --i E:\meas_MID86_ute_tra_TE100us_1H_1SL20mm_FID8353.dat --verbose --regrid=rad --vis=True 

import os,sys
import argparse
import numpy as np
from ReadRawData import ReadSiemensSpectro, ReadFastSiemensRAD, ReadFastSiemensTPI_MIA_MonoEcho, ReadFastSiemensTPI_MultiEcho
from Regridding import KaiserBesselRegridding, KaiserBesselTPI, KaiserBesselTPI_ME, Direct_Reconstruction_2D, Direct_Reconstruction_3D, KaiserBesselTPI_ME_B0, Conjuguate_Phase_DemodData, Conjuguate_Phase_combine, Display_info
from ProcessCorrections import Fieldmap_to_Source_space, Fieldmap_get, T2starestimate
from nifty_funclib import SaveArrayAsNIfTI
#from nipy import save_image
#from ctypes import *

#STD_OUTPUT_HANDLE_ID = c_ulong(0xfffffff5)
#windll.Kernel32.GetStdHandle.restype = c_ulong
#std_output_hdl = windll.Kernel32.GetStdHandle(STD_OUTPUT_HANDLE_ID)

#--i /neurospin/ciclops/projects/BIPLi7/ClinicalData/Raw_Data/2018_06_08/twix7T/meas_MID34_7Li_TPI_fisp_TR200_21deg_P05_5echos_FID2524.dat --NSTPI --s --FISTA_CSV --o /volatile/temp/test.nii
#--i /neurospin/ciclops/projects/BIPLi7/ClinicalData/Raw_Data/2017_02_21/twix7T/meas_MID203_7Li_TPI_fisp_TR200_20deg_P05_FID6746.dat --NSTPI --s --FISTA_CSV --fieldmap /neurospin/ciclops/projects/BIPLi7/ClinicalData/Processed_Data/2017_02_21/Field_mapping_2/field_mapping_phase.nii --o /neurospin/ciclops/people/Jacques/Bipli/B0Map_inhomogeneity_testzone/2017_02_21/reconstruc_fieldmap_test1.nii
#--i /neurospin/ciclops/projects/BIPLi7/ClinicalData/Raw_Data/2017_02_21/twix7T/meas_MID203_7Li_TPI_fisp_TR200_20deg_P05_FID6746.dat --NSTPI --s --FISTA_CSV --fieldmap /neurospin/ciclops/projects/BIPLi7/ClinicalData/Processed_Data/2017_02_21/Field_mapping_2/rfield_mapping_phase.nii --o /neurospin/ciclops/people/Jacques/Bipli/B0Map_inhomogeneity_testzone/2017_02_21/reconstructsTPItests/reconstruc_fieldmap_test1.nii
#--i /neurospin/ciclops/projects/SIMBA/Clinicaldata/Raw_Data/2018_08_01/twix7T/meas_MID162_23Na_TPI_TR120_FA90_P05_4mm_4echoes_FID6447.dat --NSTPI --s --FISTA_CSV --o /neurospin/ciclops/projects/SIMBA/Clinicaldata/Processed_Data/2018_08_01/TPI/Reconstruct_gridding/01-Raw/meas162.nii
#--i /neurospin/ciclops/projects/SIMBA/Clinicaldata/Raw_Data/2018_08_01/twix7T/meas_MID165_23Na_TPI_TR120_FA90_P05_4mm_4echoes_FID6450.dat --NSTPI --s --FISTA_CSV --o /neurospin/ciclops/projects/SIMBA/Clinicaldata/Processed_Data/2018_08_01/TPI/Reconstruct_gridding/01-Raw/meas165.nii
#--i V:\projects\BIPLi7\QC\twix\2018_11_20\meas_MID138_7Li_TPI_fisp_TR200_21deg_P05_5echos_FID2760.dat --FISTA_CSV --s --o V:\projects\BIPLi7\QC\dicom\2018_11_20\TPI\QC_TPI.nii
#--i V:\projects\BIPLi7\Tests\2018_10_25\meas_MID159_7Li_TPI_fisp_TR200_21deg_P05_5echos_FID684.dat --FISTA_CSV --s --o V:\projects\BIPLi7\Tests\2018_10_25\QC_TPI.nii

import time

start = time.time()

parser = argparse.ArgumentParser(epilog="RawDaRec (RawDataReconstruction) version 1.0")
parser.add_argument("--v","--verbose", help="output verbosity", action="store_true")
parser.add_argument("--i", type=str,help="Input file path")
parser.add_argument("--o", type=str,help="Output file path and name (as NIfTI)")
parser.add_argument("--vis",help="enable Visualization",action="store_true")
parser.add_argument("--regrid",type=str,help="Available trajectories : RAD")
parser.add_argument("--s","--save",help="Save Reconstructed image (with --o)",action="store_true")
parser.add_argument("--Interp",type=str,help="Interpolation Type for Radial Regridding (linear or cubic NOT nearest)")
parser.add_argument("--HeaderOnly",help="Only print Header information",action="store_true")
parser.add_argument("--TPI",help="Sandro's TPI trajectory ",action="store_true")
parser.add_argument("--Cart3D",help="3D Cartesian Trajectory",action="store_true")
parser.add_argument("--NSTPI",help="NS TPI trajectory ",action="store_true")
parser.add_argument("--Bydder",help="Enable Bydder Multicoil Reconstruction",action="store_true")
parser.add_argument("--ManualBW",help="Manualy specify BW for 2D UTE",action="store_true")
parser.add_argument("--SavePhase",help="Save Phase Image",action="store_true")
parser.add_argument("--FISTAfile", help = "Create FISTA compatible file", action="store_true")
parser.add_argument("--S", "--Spectre", help = "Process Spectroscopy Data", action="store_true")
parser.add_argument("--PSF_ND", help="Compute Point Spread Function  with NO DECAY", action="store_true")
parser.add_argument("--PSF_D", help="Compute PSF with T2 decay", action ="store_true")
parser.add_argument("--B1sensitivity", help="Compute Sensitivity map by massive subsampling of acquisition", action = "store_true")
parser.add_argument("--FISTA_CSV", help="Save Data as FISTA readable CSV Files", action = "store_true")
parser.add_argument("--SaveKspace", help="Export regridded Kspace", action = "store_true")
parser.add_argument("--UseFullDCF", help="Use Full DCF Computation --> LONG", action = "store_true")
parser.add_argument("--SumEchoes", help="Sum all echoes K space", action = "store_true")
parser.add_argument("--MIA", help="Metabolic Interleaved Acquisition", action = "store_true")
parser.add_argument("--fieldmap", type=str, help="Field Map acquisition for application of B0 correction")
parser.add_argument("--fieldcalc", type=str, help="Field Map acquisition for application of B0 correction")
args = parser.parse_args()

if args.v: verbose=True
else: verbose = False
if args.vis: vis=True
else: vis=False
if not args.i :
    #windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 12)
    print('ERROR   : Input file not specified')
    #windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
    sys.exit()
if args.i : source_file = args.i
if args.regrid: 
    regrid= True
    regridding=str(args.regrid)
    if regridding.upper() == 'RAD':
        rad=True
    else :
        rad=False    
else: 
    regrid = False
    regridding=None
    Cartesian=True
    rad=False
if args.s and args.o:
    if args.o.lower().endswith('.nii'):
        Save=True
        OutputPath=args.o
    elif args.o.lower().endswith('\\'):
        Save=True
        OutputPath=os.path.join(args.o,os.path.basename(source_file))
    else:
        Save=False
        OutputPath=None
else:
    Save=False
    OutputPath=None
if args.Interp:
    Interpolation=str(args.Interp)
else:
    Interpolation='linear'
    
if args.HeaderOnly:
    HeaderOnly=True
else:
    HeaderOnly=False
if args.TPI:
    TPI=True
else:
    TPI=False
if args.NSTPI:
    TPI=True
    NS_TPI=True
else:
    NS_TPI=False
    
if args.Bydder:
    Bydder=True
else : 
    Bydder=False
if args.SavePhase:
    SavePhase=True
else:
    SavePhase=False
if args.Cart3D:
    Cart3D=True
else:
    Cart3D=False    
if args.FISTAfile:
    FISTAfile=True
else:
    FISTAfile=False
    
if args.S:
    Spectro=True
else:
    Spectro=False
if args.PSF_ND:
    PSF_ND=True
else :
    PSF_ND=False
    
if args.PSF_D:
    PSF_D=True
else :
    PSF_D=False    
    
if args.B1sensitivity:
    B1sensitivity=True
else :
    B1sensitivity=False

if args.FISTA_CSV:
    FISTA_CSV=True
else :
    FISTA_CSV=False
    
if args.SaveKspace:
    SaveKspace=True
else :
    SaveKspace=False

if args.UseFullDCF:
    UseFullDCF=True
else :
    UseFullDCF=False    

if args.MIA:
    MIA=True
else :
    MIA=False    

if args.SumEchoes:
    SumEchoes=True
else :
    SumEchoes=False    
    
if args.fieldmap: 
    B0correct=True
    fieldmap_file=args.fieldmap
else:
    B0correct=False
    
if args.fieldcalc:
    B0correct=True
    field_interpol=True
    fieldmap_file_orig=args.fieldcalc
else:
    field_interpol=False
    
print()
print('------------------------------------------------------------')
print('Processing Pipeline :')
if HeaderOnly and not Spectro:
    print('\t\t\t => Extracting Header Information Only')
if Spectro:
    print('\t\t\t => Read Raw Data')
    print('\t\t\t => Get Spectro Data')
else:
    print('\t\t\t => Read Raw Data')
    if regrid : print('\t\t\t => regridding : ',regridding)
    print('\t\t\t => Fourier Transform')
    if vis: print('\t\t\t => Visualization')
    if FISTA_CSV : print('\t\t\t => Export FISTA readable CSV')
    if Save: print('\t\t\t => Saving Output Images !')
    else: 
        #windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 12)
        print('\t\t\t => Reconstructed Images WON\'T be saved !')
        #windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)

# if not TPI : CPLX, parameters = ExtractDataFromRawData(source_file,verbose,vis,HeaderOnly)
if not TPI and not Spectro: CPLX, parameters = ReadFastSiemensRAD(source_file,HeaderOnly)
# if TPI : CPLX, parameters,KX,KY,KZ = ExtractDataFromRawData_TPI(source_file,verbose,vis,HeaderOnly)
if TPI : 

    if MIA :
        CPLX, parameters,KX,KY,KZ = ReadFastSiemensTPI_MIA_MonoEcho(source_file,HeaderOnly,NS_TPI)
    else :    
        CPLX, parameters,KX,KY,KZ = ReadFastSiemensTPI_MultiEcho(source_file,HeaderOnly,NS_TPI)
    # Sum of averages if performed now to be correctly writen for FISTA file
    # CPLX=numpy.sum(CPLX[:,:,:,:],0);
    if CPLX.shape[0] != 1:
        CPLX=np.sum(CPLX[:,:,:,:,:],0)
    elif CPLX.shape[0] == 1 and CPLX.shape[3] != 1:
        CPLX = np.reshape(CPLX,(CPLX.shape[1],CPLX.shape[2],CPLX.shape[3],CPLX.shape[4]))        
        #CPLX = np.squeeze(CPLX)
    elif CPLX.shape[0] == 1 and CPLX.shape[3] == 1:
        CPLX = np.reshape(CPLX,(CPLX.shape[1],CPLX.shape[2],CPLX.shape[3],CPLX.shape[4]))    
    print(KX.shape)
    print(CPLX.shape)
    print("----------------------------------")
    
T2calc=0
if T2calc:
    T2starestimate(CPLX,parameters[25])
    
#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 10)
if HeaderOnly : print('Header Information Extracted')
else : 
    if not Spectro : print('Data and parameters extracted')
#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)

# KaiserBesselRegridding(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],CPLX)

# Get pixel dimensions : 
if regrid and regridding.upper() == 'RAD' :
    pix_x=float(float(parameters[8])/(float(parameters[1])*float(parameters[5])*2)) #(= FOV_x / (nbpts (*oversamplingFactor)*2))
    pix_y=float(float(parameters[9])/(float(parameters[1])*float(parameters[5])*2)) #(= FOV_y / (nbpts (*oversamplingFactor)*2) (On a deux radiales pour une dim de FOV)
else:
    pix_x=float(2*float(parameters[8])/(float(parameters[1])*float(parameters[5])))
    pix_y=float(float(parameters[9])/(float(parameters[0]))) #(= FOV_y / nbLines (EN CARTESIEN) (NO oversampling in this direction))
if TPI : 
    pix_x=float(parameters[21])
    pix_y=float(parameters[21])
    pix_z=float(parameters[21])
    affine = np.diag([-pix_x,pix_y, pix_z, 1]);
    new_affine=affine
    size=parameters[8]/parameters[21]
    source_shape=(int(size),int(size),int(size))
    for i in range(0,3):
        new_affine[i,3]=-(source_shape[i]/2-1)*affine[i,i]
    affine=new_affine
    #affine=np.array([[4., 0.,    0.,    -95], [0.,    4.,    0.,    -126], [0.,    0.,    4.,    -95], [0.,    0.,    0.,    1.]])
    
    del new_affine


if B0correct:
    freq=parameters[23]
    Timesampling=parameters[1]*parameters[26]*10**(-6)
    if field_interpol:
        fieldmap_data=Fieldmap_to_Source_space(source_shape,affine,fieldmap_file_orig)
        Hpath, Fname = os.path.split(str(OutputPath))
        OutputPath = os.path.join( Hpath + '/' + Fname[0] + '_fieldmap.nii')
        #SaveArrayAsNIfTI(Reconstruct_multiplier*ReconstructedImg[0,:,:,:],pix_x,pix_y,pix_z,OutputPath)
        SaveArrayAsNIfTI(fieldmap_data,affine,OutputPath)
        fieldmap_file=OutputPath
    else:
        fieldmap_data=Fieldmap_get(fieldmap_file)
    Nucleus=parameters[6]    
    GammaH = 42.576e6
    if Nucleus.find("1H")>-1:
        Gamma = 42.576e6
        ratio=1            
    elif Nucleus.find("23Na")>-1:
        Gamma=11.262e6
        ratio=Gamma/GammaH
    elif Nucleus.find("31P")>-1:
        Gamma= 17.235e6
        ratio=Gamma/GammaH            
    elif Nucleus.find("7Li")>-1:
        Gamma = 16.546e6
        ratio=Gamma/GammaH        
    fieldmap_data=fieldmap_data*ratio     
    diff_freq=np.max(fieldmap_data)-np.min(fieldmap_data)
    from math import ceil
    minL=ceil((4*diff_freq*Timesampling)/np.pi)
    if np.int(minL)%2==1:
        minL=minL+1
    #select optimal L => L=findL(minL,diff_freq)
    recon_method='fsc'
    ba='before'
    L=16 #because that causes deltaw to have a value close to 0, will have to mess with that down the line
   
    deltaw = np.linspace(np.max(fieldmap_data), np.min(fieldmap_data), L)*2*np.pi
    #NbProjections=parameters[18] NbPoints=parameters[1] NbAverages=parameters[2] 
    #NbCoils=parameters[3] OverSamplingFactor=parameters[5]
    #Nucleus=parameters[6] MagneticField=parameters[7] SubSampling=False 
    #CPLX=Data pval=parameters[20] resolution=parameters[21] FOV=parameters[8]
    #echoes=int(parameters[24]) TEs=parameters[25]
    coilstart=Display_info(parameters[18],parameters[1],parameters[2],parameters[3],parameters[6],parameters[7],False,KX,KY,KZ,parameters[25],fieldmap_file,Timesampling,L)
    #(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity,int(parameters[24]),SaveKspace)
    #(NbProjections,NbPoints,NbAverages,NbCoils,OverSamplingFactor,Nucleus,MagneticField,SubSampling,Data,KX,KY,KZ,pval,resolution,FOV,verbose,PSF_ND,PSF_D,B1sensitivity,echoes,SaveKspace):
    size=round(parameters[8]/parameters[21])   
    Images_L=np.zeros(shape=(parameters[24],size,size,size,L),dtype=np.complex64)
    
    
    for freq in range(L): #for freq in range(L):
        print(freq)
        if (B1sensitivity): 
            sousech=0.95
        else : sousech=0.0
        if parameters[3]:
            if parameters[6] != "1H":
                coilstart=1
            else :
                coilstart=0
        else:
            coilstart=0
        if TPI:     
            if NS_TPI:
                    DemodData_freq, ReconstructedImg_freq, Phase, Abs_Sum_of_Regridded_kspace=Conjuguate_Phase_DemodData(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity,int(parameters[24]),parameters[25],SaveKspace,fieldmap_data,Timesampling,coilstart,deltaw[freq])                                                                                                                                  #     (NbProjections,NbPoints,NbAverages,NbCoils,OverSamplingFactor,Nucleus,MagneticField,SubSampling,Data,KX,KY,KZ,pval,resolution,FOV,verbose,PSF_ND,PSF_D,B1sensitivity,echoes,TEs,SaveKspace,field_map,Timesampling,coilstart,deltaw)
            else:
                print('Error => None Neurospin TPI with Conjuguate Phase reconstruction not implemented')
        else:
            print('Error => None TPI with Conjuguate Phase reconstruction not implemented')
        if FISTA_CSV:        
            for echo in range(parameters[24]):
                FISTApath = os.path.dirname(OutputPath)
                OutputName= os.path.basename(OutputPath)
                OutputName = os.path.splitext(OutputName)            
                OutputName = OutputName[0]
                #OutputPath = os.path.join( FISTApath, 'FISTA_KspaceValues_Echo%s.csv' %echo )
                #FISTApath = os.path.dirname(OutputPath)
                #FISTApath = os.path.join( FISTApath, 'FISTA_KspaceValues_Echo%s.csv' %echo )
                FISTAPath = os.path.join( FISTApath, OutputName+'_KspaceVals_freq{0}_Echo{1}_TE_{2}.csv'.format(freq+1,echo,parameters[25][echo]) )
                #print 'Hello, this is FISTApath,',FISTApath[0]
                if not os.path.isfile(FISTAPath) :
                    # f=open("TPI_Kspace_positions_values_31P_bouleFawziTubes_p075_40960proj_FA10_TE4_5.csv","w")
                    print('INFO    : Writing CSV file for echo ', echo+1)
                    f=open(FISTAPath,"w")
                        #####f.write(float(KX))
                        ### en changeant l'incrément de la boucle on retourne plus ou moins de points de la trajectoire ! faire attention !!
                        # for i in range(0,KX.shape[0],100):
                        # echo = 0;
                        #print 'Hello, this is CPLX shape ,',CPLX.shape[0]  
                        #print 'Hello, this is KX shape 1,',KX.shape[0]  
                        #print 'Hello, this is KX shape 2,',KX.shape[1]  
                        #print 'Hello, this is an example of KX[0,0]',KX[0,0]  
                        #print 'Hello, this is an example of KY[0,0]',KY[0,0]
                        #print 'Hello, this is an example of KZ[0,0]',KZ[0,0]    
                        #print 'Hello, this is an example of np CPLX',CPLX[0,0,0,0]
                    print(KX.shape)
                    print(DemodData_freq.shape)                
                    for coil in range(DemodData_freq.shape[0]):
                        # for coil in range(1):
                        for i in range(0,KX.shape[0]):
                                # for j in range(0,KX.shape[1],10):
                                for j in range(0,KX.shape[1]):
                                    f.write(str(coil))
                                    f.write(',')
                                    f.write(str(i))
                                    f.write(',')
                                    f.write(str(j))
                                    f.write(',')
                                    f.write(str(float(KX[i,j])))
                                    f.write(',')
                                    f.write(str(float(KY[i,j])))
                                    f.write(',')
                                    f.write(str(float(KZ[i,j])))
                                    f.write(',')
                                    f.write(str(np.real(DemodData_freq[coil,i,echo,j])))
                                    f.write(',')
                                    f.write(str(np.imag(DemodData_freq[coil,i,echo,j])))
                                    f.write("\n")
                    f.close()    
        DemodData_freq=None 
            
        #Coil_Combined_Kspace_Module[echo,:,:,:,freq]=ReconstructedImg_freq[echo,:,:,:]
        for echo in range(parameters[24]):
            Images_L[echo,:,:,:,freq]=ReconstructedImg_freq[echo,:,:,:]
    #size=round (FOV/resolution)
    
    if not HeaderOnly:
        ReconstructedImg=Conjuguate_Phase_combine(Images_L,int(parameters[24]),fieldmap_data,L,size,recon_method,Timesampling,ba)
            
else: 

    if FISTA_CSV:        
        for echo in range(parameters[24]):
            FISTApath = os.path.dirname(OutputPath)
            OutputName= os.path.basename(OutputPath)
            OutputName = os.path.splitext(OutputName)            
            OutputName = OutputName[0]
            #OutputPath = os.path.join( FISTApath, 'FISTA_KspaceValues_Echo%s.csv' %echo )
            #FISTApath = os.path.dirname(OutputPath)
            #FISTApath = os.path.join( FISTApath, 'FISTA_KspaceValues_Echo%s.csv' %echo )
            FISTAPath = os.path.join( FISTApath, OutputName+'_KspaceVals_Echo{0}_TE_{1}.csv'.format(echo,parameters[25][echo]) )
            #print 'Hello, this is FISTApath,',FISTApath[0]
            if not os.path.isfile(FISTAPath) :
                # f=open("TPI_Kspace_positions_values_31P_bouleFawziTubes_p075_40960proj_FA10_TE4_5.csv","w")
                print('INFO    : Writing CSV file for echo ', echo+1)
                f=open(FISTAPath,"w")
                    #####f.write(float(KX))
                    ### en changeant l'incrément de la boucle on retourne plus ou moins de points de la trajectoire ! faire attention !!
                    # for i in range(0,KX.shape[0],100):
                    # echo = 0;
                    #print 'Hello, this is CPLX shape ,',CPLX.shape[0]  
                    #print 'Hello, this is KX shape 1,',KX.shape[0]  
                    #print 'Hello, this is KX shape 2,',KX.shape[1]  
                    #print 'Hello, this is an example of KX[0,0]',KX[0,0]  
                    #print 'Hello, this is an example of KY[0,0]',KY[0,0]
                    #print 'Hello, this is an example of KZ[0,0]',KZ[0,0]    
                    #print 'Hello, this is an example of np CPLX',CPLX[0,0,0,0]
                print(KX.shape)
                print(CPLX.shape)                
                for coil in range(CPLX.shape[0]):
                    # for coil in range(1):
                    for i in range(0,KX.shape[0]):
                            # for j in range(0,KX.shape[1],10):
                            for j in range(0,KX.shape[1]):
                                f.write(str(coil))
                                f.write(',')
                                f.write(str(i))
                                f.write(',')
                                f.write(str(j))
                                f.write(',')
                                f.write(str(float(KX[i,j])))
                                f.write(',')
                                f.write(str(float(KY[i,j])))
                                f.write(',')
                                f.write(str(float(KZ[i,j])))
                                f.write(',')
                                f.write(str(np.real(CPLX[coil,i,echo,j])))
                                f.write(',')
                                f.write(str(np.imag(CPLX[coil,i,echo,j])))
                                f.write("\n")
                f.close()  

    if Spectro:
        Spectrum = ReadSiemensSpectro(source_file,verbose)
    print('say hi')
    if not HeaderOnly:
        print('say hi again')
        if regridding and rad :
            if parameters[15] and parameters[16] and not args.ManualBW:
                # ReconstructedImg=Radial_Regridding(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],False,CPLX,Interpolation,verbose,parameters[15],parameters[16])
                ReconstructedImg,Phase=KaiserBesselRegridding(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],False,CPLX,Interpolation,verbose,parameters[15],parameters[16])
            else :
                # ReconstructedImg=Radial_Regridding(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],False,CPLX,Interpolation,verbose)
                ReconstructedImg,Phase=KaiserBesselRegridding(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],False,CPLX,Interpolation,verbose)                                                                                            #(NbProjections,NbPoints,NbAverages,NbCoils,OverSamplingFactor,Nucleus,MagneticField,SubSampling,Data,KX,KY,KZ,pval,resolution,FOV,verbose,PSF_ND,PSF_D,B1sensitivity,echoes,TEs,SaveKspace,field_map,Timesampling,deltaw):
        if TPI:  
            print('and again')
            if NS_TPI:
                print('one more time!')
                if MIA : parameters[24]=3
                # ReconstructedImg=KaiserBesselTPI(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity)
                # if SaveKspace : ReconstructedImg, Phase, Kspace =KaiserBesselTPI_ME(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity,int(parameters[24]),SaveKspace,UseFullDCF)
                if SaveKspace : ReconstructedImg, Phase, Kspace =KaiserBesselTPI_ME(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity,int(parameters[24]),SaveKspace)
                # if SavePhase and not SaveKspace : ReconstructedImg, Phase =KaiserBesselTPI_ME(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity,int(parameters[24]),SaveKspace,UseFullDCF)
                if SavePhase and not SaveKspace : ReconstructedImg, Phase =KaiserBesselTPI_ME(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity,int(parameters[24]),SaveKspace)
                else : 
                    print('tada!')
                    ReconstructedImg, Phase, Abs_Sum_of_Regridded_kspace =KaiserBesselTPI_ME(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity,int(parameters[24]),SaveKspace)
                    #ReconstructedImg, Phase, Abs_Sum_of_Regridded_kspace =KaiserBesselTPI_ME_B0(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity,int(parameters[24]),parameters[25],SaveKspace,fieldmap_data,Timesampling,L,recon_method,ba)
                    del(Phase)  
            else:
                ReconstructedImg=KaiserBesselTPI(20000,parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity)
        
        if not TPI and not regrid and Cartesian and not Cart3D and not Spectro:
            ReconstructedImg=Direct_Reconstruction_2D(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],False,CPLX)
        if not TPI and not regrid and Cart3D:
            ReconstructedImg=Direct_Reconstruction_3D(parameters[0],parameters[1],parameters[2],parameters[3],parameters[19],parameters[5],parameters[6],parameters[7],False,CPLX)

if not HeaderOnly:         
    if Save:    
        
        Reconstruct_multiplier=10**6
        if not TPI : 
            
            SaveArrayAsNIfTI(Reconstruct_multiplier*ReconstructedImg,pix_x,pix_y,float(parameters[10]),OutputPath)
            # SaveArrayAsNIfTI_2(ReconstructedImg,pix_x,pix_y,float(parameters[10]),NbPoints,NbLines,NbSlices,rad,orientation,OutputPath)
    
            # SaveArrayAsNIfTI_2(ReconstructedImg,pix_x,pix_y,float(parameters[10]),int((parameters[1])*(parameters[5])*2),int(parameters[0]),int(parameters[4]),rad,str(parameters[12]),OutputPath)
        if TPI :
            if MIA : parameters[24]=3
            if int(parameters[24])==1:
                Hpath, Fname = os.path.split(str(OutputPath))
                Fname = Fname.split('.')
                if SavePhase :
                    OutputPath = os.path.join( Hpath + '/' + Fname[0] + '_KBgrid_MODULE_Echo{0}_TE{1}.nii'.format(echo,parameters[25][echo] ))
                    #SaveArrayAsNIfTI(Reconstruct_multiplier*ReconstructedImg[0,:,:,:],pix_x,pix_y,pix_z,OutputPath)
                    SaveArrayAsNIfTI(Reconstruct_multiplier*ReconstructedImg[0,:,:,:],affine,OutputPath)
                    OutputPath = os.path.join( Hpath + '/' + Fname[0] + '_KBgrid_PHASE_Echo{0}_TE{1}.nii'.format(echo,parameters[25][echo] ))
                    #SaveArrayAsNIfTI(Phase[0,:,:,:],affine,OutputPath)
                else :
                    OutputPath = os.path.join( Hpath + '/' + Fname[0] + '_KBgrid_MODULE_Echo{0}_TE{1}.nii'.format(echo,parameters[25][echo] ))
                    SaveArrayAsNIfTI(Reconstruct_multiplier*ReconstructedImg[0,:,:,:],affine,OutputPath)
                    
                if SaveKspace :
                    OutputPath = os.path.join( Hpath + '/' + Fname[0] + '_KB_GriddedKspace_Echo{0}_TE{1}.nii'.format(echo,parameters[25][echo] ))
                    SaveArrayAsNIfTI(Kspace[0,0,:,:,:],affine,OutputPath)

            if int(parameters[24])>1:
                print("INFO    : Saving Multiple Images")
                Hpath, Fname = os.path.split(str(OutputPath))
                Fname = Fname.split('.')
                for echo in range(int(parameters[24])):
                    if SavePhase :
                        OutputPath = os.path.join( Hpath + '/' + Fname[0] + "_KBgrid_MODULE_Echo{0}_TE{1}.nii".format(echo,parameters[25][echo] ))
                        SaveArrayAsNIfTI(Reconstruct_multiplier*ReconstructedImg[echo,:,:,:],affine,OutputPath)
                        OutputPath = os.path.join( Hpath + '/' + Fname[0] + "_KBgrid_PHASE_Echo{0}_TE{1}.nii".format(echo,parameters[25][echo] ))
                        SaveArrayAsNIfTI(Phase[echo,:,:,:],affine,OutputPath)
                    else:
                        OutputPath = os.path.join( Hpath + '/' + Fname[0] + "_KBgrid_MODULE_Echo{0}_TE{1}.nii".format(echo,parameters[25][echo] ))
                        SaveArrayAsNIfTI(Reconstruct_multiplier*ReconstructedImg[echo,:,:,:],affine,OutputPath)
                if SumEchoes:
                    OutputPath = os.path.join( Hpath + '/' + Fname[0] + "_KBgrid_SumCplx_Echoes.nii")
                    SaveArrayAsNIfTI(Reconstruct_multiplier*ReconstructedImg[echo,:,:,:],affine,OutputPath)
                    # if SaveKspace :
                        # Hpath, Fname = os.path.split(OutputPath)
                        # Fname = Fname.split('.')
                        # OutputPath = os.path.join( str(Hpath), "CPLX_Kspace_Echo%s.nii" %echo )
                        # SaveArrayAsNIfTI(Kspace[echo,,:,:,:],pix_x,pix_y,pix_z,OutputPath)
    else:
        if not Spectro:
            #windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 12)
            print('WARNING : Image not saved')
            print('ERROR   : WRONG EXTENSION (should be .nii)')
            #windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
        
end = time.time()
print("INFO    : Computation time =", end - start)
