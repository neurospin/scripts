# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:44:01 2017

@author: JS247994
"""

import nibabel as nib
import numpy as np, os#, sys
from scipy import stats
from visualization import PlotReconstructedImage
import argparse,sys

#def ComputeXDensity3D(FA1,Img_FA1_path,FA2,Img_FA2_path,mask,RatioMap=0):
#RatioMap='C:\\Users\\js247994\\Documents\\Bipli2\\2017_05_16\\TPI_AC\\FA10\\T1-3D.nii'

parser = argparse.ArgumentParser(epilog="ComputeDensity version 1.0")

parser.add_argument("--v","--verbose", help="output verbosity", action="store_true")
parser.add_argument("--i1", type=str,help="Input first file path")
parser.add_argument("--i2", type=str,help="Input second file path")
parser.add_argument("--deg1", type=int,help="Flip angle of first file")
parser.add_argument("--deg2", type=int,help="Flip angle of second file")
parser.add_argument("--o", type=str,help="Output file path and name (as NIfTI)")
parser.add_argument("--m", type=str,help="Possible mask path")
parser.add_argument("--t1", type=float, help="Overall T1 value (in seconds)")
parser.add_argument("--B0map", type=str, help="Input B0 map path (if absent, set to 1 everywhere)")

args = parser.parse_args()

if args.v: verbose=True
else: verbose = False
if not args.i1 :
    #windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 12)
    print('ERROR   : Input file not specified')
    #windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
    sys.exit()
if args.i1 :
    Img_FA1_path = args.i1
    Img_FA1=nib.load(Img_FA1_path)
    Img_FA1=Img_FA1.get_data()
if not args.i2 :
    #windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 12)
    print('ERROR   : Input file not specified')
    #windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
    sys.exit()
if args.i2 :
    Img_FA2_path = args.i2
    Img_FA2 = nib.load(Img_FA2_path)
    Img_FA2 =Img_FA2.get_data()    
if not args.deg1 or not args.deg2: 
    print('Error    : Missing flip angle')
if args.deg1: degval1=args.deg1
if args.deg2: degval2=args.deg2
if args.t1: T1spec=args.t1
if not args.t1: T1spec=4.56
if args.B0map:
    RatioMap = nib.load(args.B0map) #Can be supposed as a 1 matrix (to add)
    RatioMap =RatioMap.get_data() 
if not args.B0map:    
    RatioMap=np.ones(np.shape(Img_FA1))
if args.m:
    mask= nib.loag(args.m)
    mask= mask.get_data()
if not args.m:
    mask=RatioMap
if args.o:
    outputnii=args.o
    
def degtorad(angle):
    return(angle*np.pi/180)    
    
def rval(E1,alpha,E2):
    parray=1-E1*np.cos(alpha)-E2*E2*(E1-np.cos(alpha))
    qarray=E2*(1-E1)*(1+np.cos(alpha))
    return((1-E2*E2)/(np.sqrt(parray*parray-qarray*qarray)))  
    
alphas=np.linspace(0,180,181)    
def plotT1difs(T1s,T2):
    #T1s=np.linspace(0.0001,20,20000)
    i=0
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
    
T1s=np.linspace(0.0001,20,20000)    
TR=0.2 #Lithium
TE=0.0003 #Lithium
res= 15 #Resolution in mm (isotropic)
T2=2.191

proports=plotT1difs

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
M0_T1spec = np.zeros(shape=Img_FA1.shape)
rho_T1FA = np.zeros(shape=Img_FA1.shape)
rho_T1spec = np.zeros(shape=Img_FA1.shape)
sumtest = np.zeros(shape=Img_FA1.shape)
rhoangleslow1low1 = np.zeros(shape=Img_FA1.shape)

def T1dev(T1):
    (-T1*np.exp(TR/T1))/() #Cheng et al 2006

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
                dT1=T1dev(T1FA[i,j,k])
                #M0_T1spec[i,j,k]=intercept/((1-E1spec)*np.exp(-TE/T2star))
                rho_T1FA[i,j,k]=M0_T1FA[i,j,k]/kval
                E1_T1FA[i,j,k]=slope      
                #rho_T1spec[i,j,k]=M0_T1spec[i,j,k]/kval
                rho_T1spec[i,j,k]=((Img_FA1[i,j,k]*((1-E1spec*E2-np.cos(FAMap1[i,j,k])*(E1spec-E2))/(np.sin(FAMap1[i,j,k])*(1-E1spec)))/kval)+(Img_FA2[i,j,k]*((1-E1spec*E2-np.cos(FAMap2[i,j,k])*(E1spec-E2))/(np.sin(FAMap2[i,j,k])*(1-E1spec)))/kval))/2
                sumtest[i,j,k]=(Img_FA1[i,j,k]+Img_FA2[i,j,k])/(2*kval)
            else :
                T1FA[i,j,k]=0
                M0_T1FA[i,j,k]=0 
                
from nifty_funclib import SaveArrayAsNIfTI
Hpath, Fname = os.path.split(str(outputnii))
Fname = Fname.split('.')
specialname=""
#specialname="midang1midang2"

OutputPathT1 = os.path.join( Hpath + '\\' + Fname[0]+"_T1-3D.nii")
OutputPathE1FA = os.path.join( Hpath + '\\' + Fname[0]+ "_E1_3D.nii")
OutputPathM0FA = os.path.join( Hpath + '\\' + Fname[0]+ "_M0_FA.nii")
OutputPathM0spec = os.path.join( Hpath + '\\' + Fname[0]+ "_M0_spec.nii")
OutputPathrho_T1FA = os.path.join( Hpath + '\\' + Fname[0]+ "_rho_FA.nii")
OutputPathrho_T1spec = os.path.join( Hpath + '\\' + Fname[0]+ "_rho_spec.nii")
OutputPathsumtest = os.path.join( Hpath + '\\' + Fname[0]+ "_sumtest.nii")
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