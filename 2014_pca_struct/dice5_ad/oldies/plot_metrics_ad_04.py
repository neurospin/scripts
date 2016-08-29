# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:03:38 2016

@author: ad247405
"""

import parsimony.functions.nesterov.tv
import pca_tv
import metrics

from brainomics import array_utils
import brainomics.cluster_gabriel as clust_utils

import dice5_data
import dice5_metrics

#############################################################################

INPUT_DIR = "/neurospin/brainomics/2014_pca_struct/dice5_ad/results"
INPUT_RESULTS_FILE = os.path.join(INPUT_DIR, "consolidated_results.csv")

OUTPUT_DIR = INPUT_DIR
OUTPUT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "summary.csv")

#pca_frob_0.5snr_0.1_=1578
#pca_frob_0.5_snr_0.5_=1549
#pca_frob_0.5_snr_1.0_=1530

#plot metrics  while TV is increased  

data = pd.read_csv(INPUT_RESULTS_FILE)
data=data[data.snr == snr]
data=data[data.global_pen == 0.01]


d1=data[data.l1_ratio == 0.001]
d2=data[data.l1_ratio == 0.01]
d3=data[data.l1_ratio == 0.1]
d4=data[data.l1_ratio == 0.500]


d1=d1.sort("tv_ratio")
d2=d2.sort("tv_ratio")
d3=d3.sort("tv_ratio")
d4=d4.sort("tv_ratio")
#
#plt.plot(d1.tv_ratio, d1.frobenius_test,"green",label=r'$\lambda_1/\lambda_2 = 0.001 $',linewidth=2)
#plt.plot(d2.tv_ratio, d2.frobenius_test,"blue",label=r'$\lambda_1/\lambda_2 = 0.01 $',linewidth=2)
#plt.plot(d3.tv_ratio, d3.frobenius_test,"orange",label=r'$\lambda_1/\lambda_2 = 0.1 $',linewidth=2)
#plt.plot(d4.tv_ratio, d4.frobenius_test,"pink",label=r'$\lambda_1/\lambda_2 = 0.5 $',linewidth=2)

plt.plot(data.tv_ratio,data.ssi,"green",label=r'$\lambda_1/\lambda_2 = 0.001 $',linewidth=2)

a = np.empty(3)
a.fill(0.0047)
plt.plot(data.tv_ratio,a,color='r',label=r'$\ Standard PCA results $',linewidth=2)
   
plt.ylabel("Frobenius test")
plt.ylim(1575,1578.5)
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
plt.draw()
plt.xlabel(r'TV ratio: $\lambda_{tv}/(\lambda_1 + \lambda_2 + \lambda_{tv})$')
plt.grid(True)
plt.legend()
plt.title(" SNR=0.1 - Global penalization = 0.1")

#filename=os.path.join((INPUT_DIR,"frobenius_snr=1.0_global=0.001bis"))
#plt.savefig(filename,format='png')

   

a = np.empty(5)
a.fill(0.9826)
plt.plot(d1.tv_ratio,a,color='r',label=r'$\ Standard PCA results $',linewidth=2)


     # Compute DICE coefficient of Standard PCA
components = np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad/results/data_100_100_1.0/results/0/pca_0.0_0.0_0.0/components.npz")
components=components['arr_0']

mask_0= np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad/results/data_100_100_1.0/mask_0.npy")
mask_1= np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad/results/data_100_100_1.0/mask_1.npy")
mask_2= np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad/results/data_100_100_1.0/mask_2.npy")


bin_components = components != 0
dices = np.zeros((3, ))

dices[0] = dice5_metrics.dice(mask_0, bin_components[:, 0])
dices[1] = dice5_metrics.dice(mask_1, bin_components[:, 1])    
dices[2] = dice5_metrics.dice(mask_2, bin_components[:, 2])
dice_mean_standard_pca=dices.mean()   

 #Dice coefficients plots
 for snr in (0.1,0.5,1):
     
    data = pd.read_csv(INPUT_RESULTS_FILE)
    data=data[data.snr == snr]
    data=data[data.global_pen ==0.01]


    d1=data[data.l1_ratio == 0.001]
    d2=data[data.l1_ratio == 0.01]
    d3=data[data.l1_ratio == 0.1]
    d4=data[data.l1_ratio == 0.500]


    d1=d1.sort("tv_ratio")
    d2=d2.sort("tv_ratio")
    d3=d3.sort("tv_ratio")
    d4=d4.sort("tv_ratio")


    plt.plot(d1.tv_ratio, d1.dice_mean,"green",label=r'$\lambda_1/\lambda_2 = 0.001 $',linewidth=2)
    plt.plot(d2.tv_ratio, d2.dice_mean,"blue",label=r'$\lambda_1/\lambda_2 = 0.01 $',linewidth=2)
    plt.plot(d3.tv_ratio, d3.dice_mean,"orange",label=r'$\lambda_1/\lambda_2 = 0.1 $',linewidth=2)
    plt.plot(d4.tv_ratio, d4.dice_mean,"pink",label=r'$\lambda_1/\lambda_2 = 0.5 $',linewidth=2)
    
#    plt.plot(d1.tv_ratio, d1.recall_mean,"green",label=r'$\lambda_1/\lambda_2 = 0.001 $',linewidth=2)
#    plt.plot(d2.tv_ratio, d2.recall_mean,"blue",label=r'$\lambda_1/\lambda_2 = 0.01 $',linewidth=2)
#    plt.plot(d3.tv_ratio, d3.recall_mean,"orange",label=r'$\lambda_1/\lambda_2 = 0.1 $',linewidth=2)
#        
    a = np.empty(5)
    a.fill(dice_mean_standard_pca)
    plt.plot(d1.tv_ratio,a,color='r',label=r'$\ Standard PCA results $',linewidth=2)
   
    plt.ylabel("Mean Dice Coefficient")
   
    ax = plt.gca()
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    plt.draw()
    plt.xlabel(r'TV ratio: $\lambda_{tv}/(\lambda_1 + \lambda_2 + \lambda_{tv})$')
    plt.grid(True)
    plt.legend()
    plt.title(" SNR=1.0 - Global penalization = 0.01")
    
    



