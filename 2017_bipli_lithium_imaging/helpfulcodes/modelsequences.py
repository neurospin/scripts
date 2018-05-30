# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:57:36 2018

@author: js247994
"""
import nibabel as nib
import numpy as np, os#, sys
import matplotlib.pyplot as plt

def degtorad(angle):
    return(angle*np.pi/180)    

def parr(E1,E2,alpharad):
    return(1-E1*np.cos(alpharad)-E2*E2*(E1-np.cos(alpharad)))
    
def qarr(E1,E2,alpharad):
    return(E2*(1-E1)*(1+np.cos(alpharad)))
    
def r(E1,E2,alpharad):
    parray=parr(E1,E2,alpharad)
    qarray=qarr(E1,E2,alpharad)
    return((1-E2*E2)/(np.sqrt(parray*parray-qarray*qarray)))


def calcbSSFP(E1,E2,alpharad):
    num=np.sqrt(E2*(1-E1)*np.sin(alpharad))
    den=1-(E1-E2)*np.cos(alpharad)-E1*E2
    return(num/den)
    
def MRIfunc(method,TR,TE,T1,T2,alpharad,betarad):
    E1=np.exp(-TR/T1)
    E2=np.exp(-TR/T2)      
    if method=='SSFP_fid':
        return(np.tan(alpharad/2)*(1-(E1-np.cos(alpharad))*r(E1,E2,alpharad)))
    elif method=='SSFP_echo':
        return(np.tan(alpharad/2)*(1-(1-E1*np.cos(alpharad))*r(E1,E2,alpharad)))
    elif method=='SPGR':
        E2star=E2
        return(((1-E1)/(1-E1*np.cos(alpharad)))*np.sin(alpharad)*E2star) ## The "E2 value given to the function MUST be E2star instead of E2
    elif method=='b_SSFP':
        num=(1-E1)*np.sin(alpharad)*np.sqrt(E2)
        d=1-(E1-E2)*np.cos(alpharad)-E1*E2
        return(num/d)        
    elif method=='SSFP_full':
        num=(1-E1)*(np.sin(alpharad))*np.sqrt(E2)*(1-E2*np.cos(betarad))
        d=((1-E1*np.cos(alpharad))*(1-E2*np.cos(betarad)))-(E2*(E1-np.cos(alpharad))*(E2-np.cos(betarad)))
        return(num/d)
    elif method=='echoshift':
        E2TE=np.exp(-TE/T2)
        np.sin(alpharad)*E2TE
    else:
        raise ValueError('Method was not recognized')
    
def bSSFP_alpha(T1,T2,alpharad):
    return(np.sin(alpharad)/(1+np.cos(alpharad)+(1-np.cos(alpharad))*(T1/T2)))

#def calcSSFP_fid(E1,alpharad,rval):
#    return(np.tan(alpharad/2)*(1-(E1-np.cos(alpharad))*rval))
    
#def calcSSFP_echo(E1,alpharad,rval):
#    return(np.tan(alpharad/2)*(1-(1-E1*np.cos(alpharad))*rval))

#def calcSPGR(E1,alpharad,E2star):
#    return(((1-E1)/(1-E1*np.cos(alpharad)))*np.sin(alpharad)*E2star)
    
#T1=840000
#T2=400000
    
T1=4075000
T2=1666000 

TRm=np.arange(10, 1000000,10)
TEm=np.arange(100, 50000,10)

#E2s=np.exp(-TR/T2s)

alphas=np.linspace(0,90,181)
alpha=21
alpharad=degtorad(alpha)
alphasrad=degtorad(alphas)
#alpharad=np.arccos(((T1/T2)-1)/((T1/T2)+1))
betarad=np.pi*0.8
#valsplot=plotforT1('SPGR',TRm,500,T1,T2,alpharad)
valsplot1=MRIfunc('b_SSFP',TRm,500,T1,T2,alpharad,betarad)
plt.plot(TRm,valsplot1, 'r')
valsplot2=MRIfunc('SSFP_full',TRm,500,T1,T2,alpharad,betarad)
plt.plot(TRm,valsplot2,'g')
valsplot3=MRIfunc('SSFP_fid',TRm,500,T1,T2,alpharad,betarad)
plt.plot(TRm,valsplot3,'b')
valsplot4=MRIfunc('SSFP_echo',TRm,500,T1,T2,alpharad,betarad)
plt.plot(TRm,valsplot4,'y')
#plt.show()

TRr=[75,200,	500, 750, 1000]
TRr= [ x*1000 for x in TRr]


#This is based on the acquisitions of 11 of May (Lithium)
SNR_sigmean_ov_meanstd_echo0=[46.43, 56.71, 70.08, 78.94, 78, 51.05, 59.64, 63.07, 67.52]
SNR_sigmean_ov_noisestd_echo0=[88.55, 119.1, 160.76, 146.60, 195, 105.89, 135.65, 146.76, 167.73]
SNR_sigmean_ov_meanstd_echo1=[53.92, 46.67, 38.45, 45.97, 48.90, 45.10, 60.45, 59.88, 67.74]
SNR_sigmean_ov_noisestd_echo1=[99.95, 81.96, 95.13, 105.15, 97.80, 82.69, 119.43, 122.98, 126.44]
SNR_sigmean_ov_meanstd_echo2=[48.63, 30.89, 21.19, 23.41, 24.14, 28.31, 36.56, 36.53, 36.31]
SNR_sigmean_ov_noisestd_echo2=[92.75, 49.09, 51.08, 56.74, 53.56, 50.41, 61.01, 64.15, 64.65]
SNR_sigmean_ov_meanstd_echo3=[39.96, 18.75, 14.07, 15, 14.86, 18.47, 21.97, 22.23, 21.18]
SNR_sigmean_ov_noisestd_echo3=[86.85, 34.48, 33.47, 34.09, 31.83, 33.86, 40.23, 42.12, 41.17]
SNR_sigmean_ov_meanstd_echo4=[37.84, 11.91, 12.36, 12.40, 12.38, 12.32, 14.31, 15.02, 14.56]
SNR_sigmean_ov_noisestd_echo4=[78.74, 26.12, 26.73, 26.45, 26.74, 25.81, 29.57, 32.84, 31.81]
SNR=SNR_sigmean_ov_noisestd_echo0
SNRforTRr=SNR[0:5]
#SNRforangler=SNR[1:2]+SNR[5:10]
SNRforTRr=(SNRforTRr/np.mean(SNRforTRr))*np.mean(valsplot3)
plt.plot(TRr,SNRforTRr,'r^')

SNR=SNR_sigmean_ov_noisestd_echo1
SNRforTRr=SNR[0:5]
#SNRforangler=SNR[1:2]+SNR[5:10]
SNRforTRr=(SNRforTRr/np.mean(SNRforTRr))*np.mean(valsplot4)
plt.plot(TRr,SNRforTRr,'g^')

SNR=SNR_sigmean_ov_noisestd_echo2
SNRforTRr=SNR[0:5]
#SNRforangler=SNR[1:2]+SNR[5:10]
SNRforTRr=(SNRforTRr/np.mean(SNRforTRr))*np.mean(valsplot4)
plt.plot(TRr,SNRforTRr,'b^')

SNR=SNR_sigmean_ov_noisestd_echo3
SNRforTRr=SNR[0:5]
#SNRforangler=SNR[1:2]+SNR[5:10]
SNRforTRr=(SNRforTRr/np.mean(SNRforTRr))*np.mean(valsplot4)
plt.plot(TRr,SNRforTRr,'y^')

SNR=SNR_sigmean_ov_noisestd_echo4
SNRforTRr=SNR[0:5]
#SNRforangler=SNR[1:2]+SNR[5:10]
SNRforTRr=(SNRforTRr/np.mean(SNRforTRr))*np.mean(valsplot4)
plt.plot(TRr,SNRforTRr,'w^')


plt.show()



TRr=[56250	,100000,	200000	,500000,	1000000]
SNR_sigmean_over_sigstd=[17.78811067,	20.3686555,	30.09584844,	33.42334589,	28.83354108]
SNR=(SNR_sigmean_over_sigstd/np.mean(SNR_sigmean_over_sigstd))*0.9*(np.mean(valsplot3))






#plt.plot(TRr,SNR,'y^')
#plt.show()

#TR=56250
#TE=500
TR=4000
#TE=10000
TE=2000
valsplotang1=MRIfunc('b_SSFP',TR,TE,T1,T2,alphasrad,betarad)
plt.plot(alphas,valsplotang1,'r')
valsplotang2=MRIfunc('SSFP_full',TR,TE,T1,T2,alphasrad,betarad)
plt.plot(alphas,valsplotang2,'g')
valsplotang3=MRIfunc('SSFP_fid',TR,TE,T1,T2,alphasrad,betarad)
plt.plot(alphas,valsplotang3,'b')
valsplotang4=MRIfunc('SSFP_echo',TR,TE,T1,T2,alphasrad,betarad)
plt.plot(alphas,valsplotang4,'y')
#plt.show()
#TRr=[56250	,100000,	200000	,500000,	1000000]


SNR=SNR_sigmean_ov_noisestd_echo0
SNRforangler=SNR[5:10]
Angler=[10,35,45,90]
SNRforangler=(SNRforangler/np.mean(SNRforangler))*np.mean(valsplotang3)
plt.plot(Angler,SNRforangler,'r^')

SNR=SNR_sigmean_ov_noisestd_echo1
SNRforangler=SNR[5:10]
Angler=[10,35,45,90]
SNRforangler=(SNRforangler/np.mean(SNRforangler))*np.mean(valsplotang3)
plt.plot(Angler,SNRforangler,'g^')

SNR=SNR_sigmean_ov_noisestd_echo2
SNRforangler=SNR[5:10]
Angler=[10,35,45,90]
SNRforangler=(SNRforangler/np.mean(SNRforangler))*np.mean(valsplotang3)
plt.plot(Angler,SNRforangler,'b^')

SNR=SNR_sigmean_ov_noisestd_echo3
SNRforangler=SNR[5:10]
Angler=[10,35,45,90]
SNRforangler=(SNRforangler/np.mean(SNRforangler))*np.mean(valsplotang3)
plt.plot(Angler,SNRforangler,'y^')

SNR=SNR_sigmean_ov_noisestd_echo4
SNRforangler=SNR[5:10]
Angler=[10,35,45,90]
SNRforangler=(SNRforangler/np.mean(SNRforangler))*np.mean(valsplotang3)
plt.plot(Angler,SNRforangler,'w^')
plt.show()





####Based on acquisitions on the 6th of April (Hydrogen!)


T1=80000
T2=99000 

TRm=np.arange(10, 1000000,10)
TEm=np.arange(100, 50000,10)

#E2s=np.exp(-TR/T2s)

alphas=np.linspace(0,90,181)
alpha=90
alpharad=degtorad(alpha)
alphasrad=degtorad(alphas)
#alpharad=np.arccos(((T1/T2)-1)/((T1/T2)+1))
betarad=np.pi*0.8
#valsplot=plotforT1('SPGR',TRm,500,T1,T2,alpharad)
valsplot1=MRIfunc('b_SSFP',TRm,500,T1,T2,alpharad,betarad)
plt.plot(TRm,valsplot1, 'r')
valsplot2=MRIfunc('SSFP_full',TRm,500,T1,T2,alpharad,betarad)
plt.plot(TRm,valsplot2,'g')
valsplot3=MRIfunc('SSFP_fid',TRm,500,T1,T2,alpharad,betarad)
plt.plot(TRm,valsplot3,'b')
valsplot4=MRIfunc('SSFP_echo',TRm,500,T1,T2,alpharad,betarad)
plt.plot(TRm,valsplot4,'y')

SNR_sigmean_over_sigstd=[17.78811067,	20.3686555,	30.09584844,	33.42334589,	28.83354108]
SNR_H=SNR_sigmean_over_sigstd
SNR_H=(SNR_H/np.mean(SNR_H))*0.9*np.mean(valsplot3)
plt.plot(TRr,SNR_H,'y^')
plt.show()





TR=999990
TE=500
E1=np.exp(-TR/T1)
E2=np.exp(-TR/T2) 
#num=(1-E1)*np.sin(alpharad)
alpharad=np.arccos(((T1/T2)-1)/((T1/T2)+1))
num=np.sqrt(E2)*(1-E1)*np.sin(alpharad)
den=1-(E1-E2)*np.cos(alpharad)-E1*E2
MRIdat=(num/den)
MRIdat_angle=bSSFP_alpha(T1,T2,alpharad)
np.sin(alpha)
np.arccos(((T1/T2)-1)/((T1/T2)+1))
MRIdat_angle=bSSFP_alpha(T1,T2,np.arccos(((T1/T2)-1)/((T1/T2)+1)))
#MRIdat=MRIfunc('b_SSFP',E1,E2,alpharad)
(1/2)*np.sqrt(T2/T1)

