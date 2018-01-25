# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:44:01 2017

@author: JS247994
"""

#def ComputeXDensity3D(FA1,Img_FA1_path,FA2,Img_FA2_path,mask,RatioMap=0):
    

#RatioMap='C:\\Users\\js247994\\Documents\\Bipli2\\2017_05_16\\TPI_AC\\FA10\\T1-3D.nii'
RatioMap=0
Img_FA1_path='C:\\Users\\js247994\\Documents\\Bipli2\\2017_05_16\\TPI_AC\\FA10\\LithiumPatient7_KBgrid_MODULE_Echo0_NLMFiltered.nii'
Img_FA2_path='C:\\Users\\js247994\\Documents\\Bipli2\\2017_05_16\\TPI_AC\\FA20\\Patient7_KBgrid_MODULE_Echo0_NLMFiltered.nii'
mask='C:\\Users\\js247994\\Documents\\Bipli2\\2017_05_16\\TPI_AC\\FA20\\masktest.nii'
FA1=10
FA2=21

Img_FA1_path='V:\projects\BIPLi7\Tests\\2017_12_01\dicomfiles\TPI\meas_7Li_TPI_TR200_10deg_KBgrid_MODULE_Echo0.nii'
Img_FA2_path='V:\projects\BIPLi7\Tests\\2017_12_01\dicomfiles\TPI\meas_7Li_TPI_TR200_21deg_KBgrid_MODULE_Echo0.nii'
mask='V:\projects\BIPLi7\Tests\\2017_12_01\dicomfiles\TPI\maskSig1.nii'

import nibabel as nib
import numpy as np, os#, sys
from scipy import stats
from visualization import PlotReconstructedImage

Img_FA1 = nib.load(Img_FA1_path)
Img_FA1 =Img_FA1.get_data() 
Img_FA2 = nib.load(Img_FA2_path)
Img_FA2 =Img_FA2.get_data()
mask = nib.load(mask)
mask =mask.get_data()
TR=0.2 #Lithium
TE=0.0003 #Lithium
res= 15 #Resolution in mm (isotropic)
T2star=0.005
kval=2.2288e-06
E2star=np.exp(-TE/T2star)

T1possib=3
E1possib=np.exp(-TR/T1possib)

TR=0.2 #Lithium
TE=0.0003 #Lithium
T2=2.2
res= 15 #Resolution in mm (isotropic)
T2star=0.012
kval=2.264665697646913e-06
#kval=6.1801e-06

E2star=np.exp(-TE/T2star)
E2=np.exp(-TR/T2)
T1possib=4.56


if RatioMap==0:
    path=Img_FA1_path
    RatioMap=np.ones(np.shape(Img_FA1))
    mask=RatioMap
else:
    path=RatioMap
    RatioMap = nib.load(RatioMap) #Can be supposed as a 1 matrix (to add)
    RatioMap =RatioMap.get_data()

FAMap1 = RatioMap*float(FA1)
FAMap2 = 0.8*RatioMap*float(FA2)

FAMap1=np.squeeze(FAMap1)
FAMap2=np.squeeze(FAMap2)
FAMap1=FAMap1*np.pi/180.0
FAMap2=FAMap2*np.pi/180.0
mask=np.squeeze(mask)
   # print FAMap1.shape
   # print FAMap2.shape
   # print mask.shape
T1 = np.zeros(shape=Img_FA1.shape)
M0 = np.zeros(shape=Img_FA1.shape)
rho = np.zeros(shape=Img_FA1.shape)
rhohyp = np.zeros(shape=Img_FA1.shape)
Sig2rev = np.zeros(shape=Img_FA1.shape)
Sig1rev = np.zeros(shape=Img_FA1.shape)

rhoangleslow1low1 = np.zeros(shape=Img_FA1.shape)

T1hyp=4.56
E1hyp=np.exp(-TR/T1hyp)
for i in range(Img_FA1.shape[0]):
    for j in range(Img_FA1.shape[1]):
        for k in range(Img_FA1.shape[2]):
            if mask[i,j,k]>0:
                y = np.zeros(2)
                x = np.zeros(2)
                y[0]=Img_FA1[i,j,k]/np.sin(FAMap1[i,j,k])
                y[1]=Img_FA2[i,j,k]/np.sin(FAMap2[i,j,k])
                x[0]=Img_FA1[i,j,k]/np.tan(FAMap1[i,j,k])
                x[1]=Img_FA2[i,j,k]/np.tan(FAMap2[i,j,k])
                slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

                if not np.isnan(-TR/np.log(slope)):
                    T1[i,j,k]=-TR/np.log(slope)
                else:
                    T1[i,j,k]=0
                M0[i,j,k]=intercept/((1-slope)*np.exp(-TE/T2star))
                rho[i,j,k]=M0[i,j,k]/kval
                rhohyp[i,j,k]=intercept/((1-E1hyp)*np.exp(-TE/T2star))
                #M01=(Img_FA1[i,j,k]*(1-slope*np.cos(FAMap1[i,j,k])))/(np.sin(FAMap1[i,j,k]))
                #M02=(Img_FA2[i,j,k]*(1-slope*np.cos(FAMap2[i,j,k])))/(np.sin(FAMap2[i,j,k]))
                #M0[i,j,k]=(M01+M02)/(2*E2star*kval*(1-E1possib))#(1-slope))
                #Sig2rev[i,j,k]=(M02*np.sin(FAMap1[i,j,k]))/(1-slope*np.cos(FAMap1[i,j,k]))
                #Sig1rev[i,j,k]=(M01*np.sin(FAMap2[i,j,k]))/(1-slope*np.cos(FAMap2[i,j,k]))
                #print(Img_FA1[i,j,k],Sig1rev[i,j,k],Img_FA2[i,j,k],Sig2rev[i,j,k])
                #print('\n')
            else :
                T1[i,j,k]=0
                M0[i,j,k]=0 
                
from nifty_funclib import SaveArrayAsNIfTI
Hpath, Fname = os.path.split(str(path))
Fname = Fname.split('.')
specialname="midang1lowang2"
OutputPathT1 = os.path.join( Hpath + '\\SPGR\\' + "T1-3D"+specialname+".nii")
OutputPathM0 = os.path.join( Hpath + '\\' + "M0-3D.nii")
OutputPathrho = os.path.join( Hpath + '\\SPGR\\' + "rho-3D"+specialname+".nii")
OutputPathrhohyp = os.path.join( Hpath + '\\' + "rhohyp-3D.nii")



#OutputPathSig1rev = os.path.join( Hpath + '\\' + "Sig1theory.nii")
#OutputPathSig2rev = os.path.join( Hpath + '\\' + "Sig2theory.nii")
SaveArrayAsNIfTI(T1,res,res,res,OutputPathT1) 
#SaveArrayAsNIfTI(M0,res,res,res,OutputPathM0) 
SaveArrayAsNIfTI(rho,res,res,res,OutputPathrho) 
#SaveArrayAsNIfTI(rhohyp,res,res,res,OutputPathrhohyp) 
#SaveArrayAsNIfTI(Sig1rev,res,res,res,OutputPathSig1rev) 
#SaveArrayAsNIfTI(Sig2rev,res,res,res,OutputPathSig2rev) 