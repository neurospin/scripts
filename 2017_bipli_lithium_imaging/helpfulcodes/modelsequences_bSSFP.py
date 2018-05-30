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
        Bw=(1000/(TR/1000-2.2))
        return(num/(d*np.sqrt(Bw)))        
    elif method=='SSFP_full':
        num=(1-E1)*(np.sin(alpharad))*np.sqrt(E2)*(1-E2*np.cos(betarad))
        d=((1-E1*np.cos(alpharad))*(1-E2*np.cos(betarad)))-(E2*(E1-np.cos(alpharad))*(E2-np.cos(betarad)))
        return(num/(d*np.sqrt(1)))
    elif method=='echoshift':
        E2TE=np.exp(-TE/T2)
        np.sin(alpharad)*E2TE
    else:
        raise ValueError('Method was not recognized')
    
def bSSFP_alpha(T1,T2,alpharad):
    return(np.sin(alpharad)/(1+np.cos(alpharad)+(1-np.cos(alpharad))*(T1/T2)))

#T1=840000
#T2=400000
      
#Lithium reference tube values
T1=4075000
T2=1666000 

TRm=np.arange(2300, 100000,1)
TEm=TRm/2

alphas=np.linspace(0,90,181)
alpha=30
alpharad=degtorad(alpha)
alphasrad=degtorad(alphas)
betarad=np.pi*0.8
valsplot1=MRIfunc('b_SSFP',TRm,TRm/2,T1,T2,alpharad,betarad)
plt.plot(TRm,valsplot1, 'r')
valsplot2=MRIfunc('SSFP_full',TRm,TRm/2,T1,T2,alpharad,betarad)/3
valsplot3=MRIfunc('SSFP_fid',TRm,TRm/2,T1,T2,alpharad,betarad)/3
valsplot4=MRIfunc('SSFP_echo',TRm,TRm/2,T1,T2,alpharad,betarad)/3

TRr=[9.55,8.53,	7.53, 6.56, 5.53, 5.06, 4.53, 4.06, 3.74, 3.54]
TEr=[4.78,4.27,3.77,3.28,2.77,2.53,2.27,2.03,1.87,1.77]
TRr= [ x*1000 for x in TRr]

#16_05_2018 acquisitions
SNR_sigmean_ov_meannoise=[49.94600582, 46.96708961, 44.36933099, 35.50937719, 31.25884877, 32.03426905, 25.96979274, 22.73443152, 22.81123929, 20.47538435]
SNR_sigmean_ov_noisestd=[88.07648573, 85.55123626, 78.5220489, 69.66991575, 55.52830539, 60.45238468, 48.78957605, 45.47586671, 43.60481586, 39.16199363]

SNR=SNR_sigmean_ov_meannoise
SNRforTRr=SNR[0:10]
SNRforTRr=(SNRforTRr/np.mean(SNRforTRr))*np.mean(valsplot1)
plt.plot(TRr,SNRforTRr,'r^')

plt.show()

TR=10000
TE=5000
valsplotang1=MRIfunc('b_SSFP',TR,TE,T1,T2,alphasrad,betarad)
plt.plot(alphas,valsplotang1,'r')
valsplotang2=MRIfunc('SSFP_full',TR,TE,T1,T2,alphasrad,betarad)
valsplotang3=MRIfunc('SSFP_fid',TR,TE,T1,T2,alphasrad,betarad)
vdghalsplotang4=MRIfunc('SSFP_echo',TR,TE,T1,T2,alphasrad,betarad)


#Values obtained on 23th of May, 2018,
angler=[10, 20, 30, 40, 45]
SNR_sigmean_ov_meannoise=[7.754483778, 16.00752634, 22.96760168, 28.55848053, 32.57717739, 12.90210387, 32.73106012]
SNR_sigmean_ov_noisestd=[14.59331803, 29.40958611, 35.56966513, 46.68337655, 48.45415887, 25.80965577, 42.56827585]

SNR=SNR_sigmean_ov_meannoise
SNRforangler=SNR[0:5]
#SNRforangler=SNR[1:2]+SNR[5:10]
SNRforangler=(SNRforangler/np.mean(SNRforangler))*np.mean(valsplotang1)
plt.plot(angler,SNRforangler,'r^')

plt.show()

#H cylinder values
T1=80000
T2=99000 

#Li clinical values
T1=3947000
T2=63000

alpha=14
alpharad=degtorad(alpha)

valsplot1=MRIfunc('b_SSFP',TRm,TRm/2,T1,T2,alpharad,betarad)
plt.plot(TRm,valsplot1, 'r')
plt.axis([2000, np.max(TRm), 0, np.max(valsplot1)+0.0002])
plt.show()

TAcq=51.04*(TRm/1000)-23.473

valsplot2=valsplot1*np.sqrt(60/TAcq)

plt.plot(TRm,valsplot2, 'b')
plt.axis([2000, np.max(TRm), 0, np.max(valsplot2)+0.0002])

alphas=np.linspace(0,90,181)
SNRMax=0
for alpha in alphas:
    alpharad=degtorad(alpha)
    valsplot1=MRIfunc('b_SSFP',TRm,TRm/2,T1,T2,alpharad,betarad)
    if np.max(valsplot1)>SNRMax:
        bestalpha=alpha
        SNRMax=np.max(valsplot1)        
        
TR=10000
TE=TR/2
alphasrad=degtorad(alphas)
valsplotangnew=MRIfunc('b_SSFP',TR,TE,T1,T2,alphasrad,betarad)
plt.plot(alphas,valsplotangnew,'g')
plt.show()

alpha=14
alpharad=degtorad(alpha)
valsplot1=MRIfunc('b_SSFP',TRm,TRm/2,T1,T2,alpharad,betarad)
plt.plot(TRm,valsplot1,'g')
plt.show()
