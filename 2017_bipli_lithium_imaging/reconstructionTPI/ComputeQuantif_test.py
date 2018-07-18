# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:17:38 2018

@author: js247994
"""

#This code serves to brin the raw signal to theconcentration values by basing itself on T1 input.

#from ComputeDensity3D_clinic import degtorad,rval
#python ComputeQuantif_clinic.py --i V:\projects\BIPLi7\ClinicalData\Processed_Data\2018_06_01\TPI\Reconstruct_gridding\01-Raw\Patient...

#codelaunchVFA=({'"'},Pythonexe+{'" '}+ComputeVFAfile+{' --i1 '}+file1path+{' --i2 '}+file2path+{' --deg1 '}+deg1+{' --deg2 '}+deg2+{' --t1 '}+ num2str(T1val)+ {' --v --o '}+Computedniipath)

import numpy as np
import nibabel as nib
from scipy import stats
import os
from reconstructionTPI.nifty_funclib import SaveArrayAsNIfTI

def degtorad(angle):
    return(angle*np.pi/180)    
    
def T1dev(T1,TR):
    (-T1*np.exp(TR/T1))/() #Cheng et al 2006

def rval(E1,alpha,E2):
    parray=1-E1*np.cos(alpha)-E2*E2*(E1-np.cos(alpha))
    qarray=E2*(1-E1)*(1+np.cos(alpha))
    return((1-E2*E2)/(np.sqrt(parray*parray-qarray*qarray)))  
        
def plotT1difs(T1s,T2,TR):
    #T1s=np.linspace(0.0001,20,20000)
    i=0
    alphas=np.linspace(0,180,181)
    proportSSFP=np.zeros(np.size(T1s))
    for T1 in T1s:
        E1=np.exp(-TR/T1)
        alpharads=alphas*np.pi/180.0    
        E2=np.exp(-TR/T2)
        rvals=rval(E1,alpharads,E2)
        valsSSFP=np.tan(alpharads/2)*(1-(E1-np.cos(alpharads))*rvals)
        #valsSPGR=(np.sin(alpharads)*(1-E1))/((1-E1*np.cos(alpharads)))   
        proportSSFP[i]=valsSSFP[10]/valsSSFP[21]
        #proportSPGR[i]=valsSPGR[10]/valsSPGR[21]
        i=i+1
    #plt.plot(T1s,proportSSFP,T1s,proportSPGR)        
    #plt.plot(T1s,proportSSFP)        
    #plt.axis([0, 20, 0, 2.5])
    return(proportSSFP)


def raw_to_concentration_simp(Img_Path,degval,T1,B0Mapfile,maskfile,outputnii,verbose):
    #values are displayed in micro-seconds (consistent with .dat values)    

####" Establishing all parameters ####
    Img=nib.load(Img_Path)
    Img=Img.get_data()   
    
    if os.path.isfile(B0Mapfile):
        RatioMap_path = nib.load(B0Mapfile) #Can be supposed as a 1 matrix (to add)
        RatioMap_path =RatioMap_path.get_data() 
    else:    
        RatioMap=np.ones(np.shape(Img))
        
    if os.path.isfile(maskfile):
        mask= nib.load(maskfile)
        mask= mask.get_data()
    else:
        mask=RatioMap   
    
##### All of these parameters should be added in a readable "param" file at some point, particularly concerning the T1/T2 and estimated kvalues        
    TR=200000 #Lithium
    TE=300 #Lithium
    res= 15 #Resolution in mm (isotropic)
    T1=3947000
    T2=63000
    T2star=12000
    #kvalSPGR=2.264665697646913e-06
    kvalSPGR=2.2113e-06
    kvalSSFP=2.2621e-06
    E2star=np.exp(-TE/T2star)
    E1=np.exp(-TR/T1)
    E2=np.exp(-TR/T2)
    
    FAMap = RatioMap*float(degval)
    
    FAMap=np.squeeze(FAMap)
    FAMap=FAMap*np.pi/180.0
    mask=np.squeeze(mask)
    
#   M0_T1SPGR = np.zeros(shape=Img.shape)
    rho_T1SPGR = np.zeros(shape=Img.shape)
    rho_T1SSFP = np.zeros(shape=Img.shape)
    
    #T1hyp=4.56
    #E1hyp=np.exp(-TR/T1hyp)
    for i in range(Img.shape[0]):
        for j in range(Img.shape[1]):
            for k in range(Img.shape[2]):
                if mask[i,j,k]>0:
                    rho_T1SPGR[i,j,k]=Img[i,j,k]*((1-E1*E2-np.cos(FAMap[i,j,k])*(E1-E2))/(np.sin(FAMap[i,j,k])*(1-E1)))/kvalSPGR
                    rho_T1SSFP[i,j,k]=Img[i,j,k]/(kvalSSFP*(np.tan(FAMap[i,j,k]/2)*(1-(E1-np.cos(FAMap[i,j,k]))*rval(E1,FAMap[i,j,k],E2))));
                    # 1/(kvalSSFP*(np.tan(degval/2)*(1-(E1spec-np.cos(degval))*rval(E1spec,degval,E2))))
    Hpath, Fname = os.path.split(str(outputnii))
    Fname = Fname.split('.')
    OutputPathrho_T1SPGR = os.path.join( Hpath,Fname[0]+"_rhoSPGR.nii")
    OutputPathrho_T1SSFP = os.path.join( Hpath,Fname[0]+"_rhoSSFP.nii")
    
    if verbose:
        print((degval,T1,E1))
    SaveArrayAsNIfTI(rho_T1SSFP,res,res,res,OutputPathrho_T1SSFP) 
    SaveArrayAsNIfTI(rho_T1SPGR,res,res,res,OutputPathrho_T1SPGR) 
    

def raw_to_concentration_VFA(Img_Path_1,degval1,Img_Path_2,degval2,T1,output,B0Mapfile,maskfile,outputnii,verbose):
    

    TR=0.2 #Lithium
    TE=0.0003 #Lithium

    T2=2.191

#    T1s=np.linspace(0.0001,20,20000)    
    res= 15 #Resolution in mm (isotropic)
#    proports=plotT1difs
    
    Img_FA1=nib.load(Img_Path_1)
    Img_FA1=Img_FA1.get_data()
    
    Img_FA2=nib.load(Img_Path_2)
    Img_FA2=Img_FA2.get_data()
    
    T1spec=T1 #4.56
    
    if os.path.isfile(B0Mapfile):
        RatioMap_path = nib.load(B0Mapfile) #Can be supposed as a 1 matrix (to add)
        RatioMap_path =RatioMap_path.get_data() 
    else:    
        RatioMap=np.ones(np.shape(Img_FA1))
        
    if os.path.isfile(maskfile):
        mask= nib.load(maskfile)
        mask= mask.get_data()
    else:
        mask=RatioMap   
    

    
    T2star=0.012
    #T2star=0.005
    #kval=2.2288e-06
    kval=2.264665697646913e-06
    #kval=6.1801e-06
    E2star=np.exp(-TE/T2star)
    #T1possib=4.56
    E1spec=np.exp(-TR/T1spec)
    E2=np.exp(-TR/T2)
    
    FAMap1 = RatioMap*float(degval1)
    FAMap2 = RatioMap*float(degval2)
    #FAMap2 = 0.8*RatioMap*float(FA2)
    
    FAMap1=np.squeeze(FAMap1)
    FAMap2=np.squeeze(FAMap2)
    FAMap1=FAMap1*np.pi/180.0
    FAMap2=FAMap2*np.pi/180.0
    mask=np.squeeze(mask)
    
    T1FA = np.zeros(shape=Img_FA1.shape)
    E1_T1FA = np.zeros(shape=Img_FA1.shape)
    M0_T1FA = np.zeros(shape=Img_FA1.shape)
#    M0_T1spec = np.zeros(shape=Img_FA1.shape)
    rho_T1FA = np.zeros(shape=Img_FA1.shape)
    rho_T1spec = np.zeros(shape=Img_FA1.shape)
    sumtest = np.zeros(shape=Img_FA1.shape)
    rhoangleslow1low1 = np.zeros(shape=Img_FA1.shape)
    

    
    #T1hyp=4.56
    #E1hyp=np.exp(-TR/T1hyp)
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
      
        
                    ######      Other SPGR calc method   (Liberman 2014)        #########
                    yt = np.zeros(2)
                    xt = np.zeros(2)
                    yt[0]=Img_FA1[i,j,k]/(1-np.exp(-TR/T1spec)*np.cos(FAMap1[i,j,k]))
                    yt[1]=Img_FA2[i,j,k]/(1-np.exp(-TR/T1spec)*np.cos(FAMap2[i,j,k]))
                    xt[0]=(Img_FA1[i,j,k]*np.cos(FAMap1[i,j,k]))/((1-np.exp(-TR/T1spec)*np.cos(FAMap1[i,j,k]))*(1-np.exp(-TR/T1spec)*np.cos(FAMap1[i,j,k])))
                    xt[1]=(Img_FA2[i,j,k]*np.cos(FAMap2[i,j,k]))/((1-np.exp(-TR/T1spec)*np.cos(FAMap2[i,j,k]))*(1-np.exp(-TR/T1spec)*np.cos(FAMap2[i,j,k])))
                    slopet, interceptt, r_valuet, p_valuet, std_errt = stats.linregress(xt,yt)
                    #######################################################                
                    
                    if not np.isnan(-TR/np.log(slope)):
                        T1FA[i,j,k]=-TR/np.log(slope)
                    else:
                        T1FA[i,j,k]=0
                    M0_T1FA[i,j,k]=intercept/((1-slope)*np.exp(-TE/T2star))
                    #M0_T1spec[i,j,k]=intercept/((1-E1spec)*np.exp(-TE/T2star))
                    rho_T1FA[i,j,k]=M0_T1FA[i,j,k]/kval
                    E1_T1FA[i,j,k]=slope      
                    #rho_T1spec[i,j,k]=M0_T1spec[i,j,k]/kval
                    rho_T1spec[i,j,k]=((Img_FA1[i,j,k]*((1-E1spec*E2-np.cos(FAMap1[i,j,k])*(E1spec-E2))/(np.sin(FAMap1[i,j,k])*(1-E1spec)))/kval)+(Img_FA2[i,j,k]*((1-E1spec*E2-np.cos(FAMap2[i,j,k])*(E1spec-E2))/(np.sin(FAMap2[i,j,k])*(1-E1spec)))/kval))/2
                    sumtest[i,j,k]=(Img_FA1[i,j,k]+Img_FA2[i,j,k])/(2*kval)
                else :
                    T1FA[i,j,k]=0
                    M0_T1FA[i,j,k]=0 
                    
    from reconstructionTPI.nifty_funclib import SaveArrayAsNIfTI
    
    Hpath, Fname = os.path.split(str(outputnii))
    Fname = Fname.split('.')
#    specialname=""
    #specialname="midang1midang2"
    
    OutputPathT1 = os.path.join( Hpath, Fname[0],"_T1-3D.nii")
    OutputPathE1FA = os.path.join( Hpath,Fname[0],"_E1_3D.nii")
    OutputPathM0FA = os.path.join( Hpath,Fname[0],"_M0_FA.nii")
    OutputPathM0spec = os.path.join( Hpath,Fname[0],"_M0_spec.nii")
    OutputPathrho_T1FA = os.path.join( Hpath,Fname[0],"_rho_FA.nii")
#    OutputPathsumtest = os.path.join( Hpath + '\\' + Fname[0]+ "_sumtest.nii")
    #OutputPathrhohyp = os.path.join( Hpath + '\\' + "rhohyp-3D2.nii")
    
    if verbose:
        print((degval1,degval2,T1spec,E1spec))
        print((OutputPathT1,OutputPathE1FA,OutputPathM0FA,OutputPathM0spec,OutputPathrho_T1FA,OutputPathrho_T1spec))
        print(Fname)
    
    SaveArrayAsNIfTI(T1FA,res,res,res,OutputPathT1) 
    #SaveArrayAsNIfTI(E1FA,res,res,res,OutputPathE1FA) 
    SaveArrayAsNIfTI(M0_T1FA,res,res,res,OutputPathM0FA)
    #SaveArrayAsNIfTI(M0_T1spec,res,res,res,OutputPathM0spec) 
    SaveArrayAsNIfTI(rho_T1FA,res,res,res,OutputPathrho_T1FA) 
    SaveArrayAsNIfTI(rho_T1spec,res,res,res,OutputPathrho_T1spec) 
#SaveArrayAsNIfTI(sumtest,res,res,res,OutputPathsumtest) 