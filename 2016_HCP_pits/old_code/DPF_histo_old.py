"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import nibabel.gifti.giftiio as gio
import os, json, time, glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib as mpl
from pylab import *
from scipy.optimize import curve_fit  
import pandas as pd
from sklearn import mixture
from sklearn.preprocessing import normalize
# The following boolean should be set to true if you want to resample the DPF in each parcel
# This should be done if you have changed the template parcel for example
# Else you save ~18min by not re-creating all the files
COLLECT_PITS_IN_EACH_PARCEL = True

OUTPUT  = '/neurospin/brainomics/2016_HCP/distribution_Depth_hcp_BV/'

#path0 = '/media/yl247234/SAMSUNG/hcp/Freesurfer_mesh_database/'
path0 = '/media/yl247234/SAMSUNG/hcp/databaseBV/'
path_parcels = '/media/yl247234/SAMSUNG/hcp/databaseBV/'
#path_parcels = '/media/yl247234/SAMSUNG/all_data/Freesurfer_mesh_database/'
path = path0+'hcp/'
temp_file_s_ids='/home/yl247234/s_ids_lists/s_ids.json'
sides = ['R', 'L']
with open(temp_file_s_ids, 'r') as f:
    data = json.load(f)
s_ids  = list(json.loads(data))
     

label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size 
text_size = 16
  
def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def plot_histo(ax, x):
    X  = x.reshape(-1, 1)
    gmm = mixture.GMM(n_components=2) # gmm for two components
    gmm.fit(X) # train it!

    COLOR, label, alpha, NORMED = 'blue', 'pits', 1, False
    number_bins = x.shape[0]/16
    a,b = min(x), max(x)
    ax2 = ax.twinx()
    n, bins, patches = ax.hist(X, number_bins, facecolor=COLOR, alpha=alpha, range=(a,b), label=label+": "+str(x.shape[0]), normed=NORMED)
    x_bins=(bins[1:]+bins[:-1])/2
    
    """
    params,cov=curve_fit(bimodal, x_bins, n, maxfev=1000000)
    ax.plot(x_bins,bimodal(x_bins,*params), color='red',lw=3,label='model')
    df = pd.DataFrame(data={'params':params},index=bimodal.__code__.co_varnames[1:])
    arbitrary = False
    arbitrary2 = False
    arbitrary3 = False
    if abs(df['params']['mu1'])< 2 and  abs(df['params']['mu2']) <2 :
        x_samples = [df['params']['mu2']+(df['params']['mu1']-df['params']['mu2'])/250.0*p for p in range(250)]
        gauss1 = gauss(x_samples, df['params']['mu1'], df['params']['sigma1'], df['params']['A1'])
        gauss2 = gauss(x_samples, df['params']['mu2'], df['params']['sigma2'], df['params']['A2'])
        threshold = x_samples[np.argmin(np.abs(gauss1-gauss2))]
    elif abs(df['params']['mu1'])< 2:
        arbitrary2 = True
        print "mu1 based"
        print df['params']['mu1']
        print df['params']['sigma1']
        threshold = df['params']['mu1']-2*abs(df['params']['sigma1'])
    elif  abs(df['params']['mu2'])< 2:
        print "mu2 based"
        threshold = df['params']['mu2']-2*abs(df['params']['sigma2'])
        arbitrary3 = True
    else:
        print "No threshold set"
        print df
        threshold = 1.0
        arbitrary = True
    if threshold < 0 and "Freesurfer" not in path:
        print "Threshold < 0 : " +str(threshold)
        print df
        threshold = 1.0
        arbitrary = True
    """
    linspace = np.linspace(-2, 2, 1000).reshape(-1, 1)
    ax2.plot(linspace, np.exp(gmm.score_samples(linspace)[0]), 'r')
    mu1 = gmm.means_[0]
    mu2 = gmm.means_[1]
    std1 = np.sqrt(gmm.covars_[0])
    std2 = np.sqrt(gmm.covars_[1])
    A1 = gmm.weights_[0]
    A2 = gmm.weights_[1]
    arbitrary = False
    arbitrary2 = False
    arbitrary3 = False
    if abs(mu1)< 2 and  abs(mu2) <2 :
        x_samples = [mu2+(mu1-mu2)/250.0*p for p in range(250)]
        gauss1 = gauss(x_samples, mu1, std1, 1)
        gauss2 = gauss(x_samples, mu2, std2, 1)
        threshold = x_samples[np.argmin(np.abs(gauss1-gauss2))]
    elif abs(df['params']['mu1'])< 2:
        arbitrary2 = True
        print "mu1 based"
        print mu1
        print std1
        threshold = mu1-2*std1
    elif  abs(mu2)< 2:
        print "mu2 based"
        print mu2
        print std2
        threshold = mu2-2*std2
        arbitrary3 = True
    else:
        print "No threshold set"
        #print df
        threshold = 1.0
        arbitrary = True
    """if threshold < 0 and "Freesurfer" not in path:
        print "Threshold < 0 : " +str(threshold)
        #print df
        threshold = 1.0
        arbitrary = True"""

    mu = np.mean(x)
    #std = np.std(x)
    if arbitrary:
        ax.plot(np.repeat(threshold,200), np.linspace(min(n), max(n), num=200), color='magenta', lw=3, label = 'threshold')
    elif arbitrary2:
        ax.plot(np.repeat(threshold,200), np.linspace(min(n), max(n), num=200), color='cyan', lw=3, label = 'threshold')
    elif arbitrary3:
        ax.plot(np.repeat(threshold,200), np.linspace(min(n), max(n), num=200), color='indigo', lw=3, label = 'threshold')
    else:
        ax.plot(np.repeat(threshold,200), np.linspace(min(n), max(n), num=200), color='green', lw=3, label = 'threshold')
    ax.set_xlim([-2,2])
    div = int(round(max(n)))/10
    ax.set_ylim([0, div*10+10])
    ax.set_title(r'$\mu$'+'1: '+str(round(mu1,2))+' '+r'$\mu$'+'2: '+str(round(mu2,2))+' '+r'$\mu$'+': '+str(round(mu,2)),fontsize =text_size-4, fontweight = 'bold',verticalalignment="bottom")
    return threshold
    
    #ax.plot(amps,Cvels,color=colorVal,  linestyle='--', marker='o', markersize = 5)
    #ax.set_xscale('log')
 
     
 
 
 

if COLLECT_PITS_IN_EACH_PARCEL:
    for side in sides:
        print "\n"
        print side
        OUTPUT_side = OUTPUT+side+'/'
        if not os.path.exists(OUTPUT_side):
            os.makedirs(OUTPUT_side)

        t0 = time.time()
        file_parcels_on_atlas = path_parcels +'pits_density/clusters_total_average_pits_smoothed0.7_60_sym_dist15.0_area100.0_ridge2.0.gii'
        array_parcels = gio.read(file_parcels_on_atlas).darrays[0].data
        parcels_org =  np.unique(array_parcels)
        parcels_org = parcels_org[1:]
        NB_PARCELS = len(parcels_org)
        for k,parcel in enumerate(parcels_org):
            t = time.time()
            X = np.array([])
            for j, s_id in enumerate(s_ids):
                file_pits_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                                  "default_analysis", "segmentation", "mesh",
                                                  "surface_analysis_sym", ""+s_id+"_"+side+"white_pits_on_atlas.gii")
                file_DPF_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                                 "default_analysis", "segmentation", "mesh",
                                                 "surface_analysis_sym", ""+s_id+"_"+side+"white_DPF_on_atlas.gii")
                if os.path.isfile(file_pits_on_atlas) and os.path.isfile(file_DPF_on_atlas):
                    array_pits = gio.read(file_pits_on_atlas).darrays[0].data
                    array_DPF = gio.read(file_DPF_on_atlas).darrays[0].data

                    index_pits = np.nonzero(array_pits)[0]
                    parcels = array_parcels[index_pits]
                    ind = np.where(parcel == parcels)[0]
                    # If the subject has pit in this parcel we consider add their position
                    if ind.size:
                        X = np.concatenate((X,array_DPF[index_pits[ind]]))

            np.savetxt(OUTPUT_side+'Parcel_'+str(parcel)+'.txt', X)
            print "Elapsed time for parcel " +str(parcel)+ " : "+ str(time.time()-t)
        print "Elapsed time for side " +str(side)+ " : "+ str(time.time()-t0)







for side in sides:
    OUTPUT_side = OUTPUT+side+'/'
    if not os.path.exists(OUTPUT_side):
        os.makedirs(OUTPUT_side)
    t0 = time.time()
    file_parcels_on_atlas = path_parcels +'pits_density/clusters_total_average_pits_smoothed0.7_60_sym_dist15.0_area100.0_ridge2.0.gii'
    array_parcels = gio.read(file_parcels_on_atlas).darrays[0].data
    parcels_org =  np.unique(array_parcels)
    parcels_org = parcels_org[1:]
    NB_PARCELS = len(parcels_org)
    count  = 0
    thresholds = []
    for k,parcel in enumerate(parcels_org):
        if not count%30:
            if k > 1:
                fig.subplots_adjust(hspace=0.2,wspace=0.2)
            count = 0
            gs = gridspec.GridSpec(5, 6, width_ratios=[8,8,8,8,8,8], height_ratios=[6,6,6,6,6])
            fig = plt.figure()
            ax = []
        count +=1
       
        ax.append(fig.add_subplot(gs[k%30]))
        x  = np.loadtxt(OUTPUT_side+'Parcel_'+str(parcel)+'.txt')
        thresholds.append(plot_histo(ax[k%30], x))
        
        ax[k%30].locator_params(axis = 'x', nbins=4)
        ax[k%30].locator_params(axis='y',nbins=5)
        ax[k%30].spines['right'].set_visible(False)
        ax[k%30].spines['top'].set_visible(False)
        ax[k%30].xaxis.set_ticks_position('none')
        ax[k%30].yaxis.set_ticks_position('none')
    

        
        ax[k%30].yaxis.set_ticks_position('left')
        if k%30%6==0:
            ax[k%30].set_ylabel('Number of pits', fontsize=text_size, fontweight = 'bold', labelpad=0)
        """
        if k%30%6==0:
            ax[k%30].yaxis.set_ticks_position('left')
            ax[k%30].set_ylabel('Number of pits', fontsize=text_size, fontweight = 'bold', labelpad=0)
        else:
            ax[k%30].spines['left'].set_visible(False)
            for ylabel_i in ax[k%30].get_yticklabels():
                ylabel_i.set_visible(False)
                ylabel_i.set_fontsize(0.0)
        """
        if k%30 < 24:
            ax[k%30].spines['bottom'].set_visible(False)
            for xlabel_i in ax[k%30].get_xticklabels():
                xlabel_i.set_visible(False)
                xlabel_i.set_fontsize(0.0)
                #ax[k%30].set_ylim([60,100])
 
        else:
            ax[k%30].xaxis.set_ticks_position('bottom')
            ax[k%30].set_xlabel('DPF', fontsize=text_size, fontweight = 'bold', labelpad=0)
            #ax[k%30].set_xlabel('Amplitude stimulus [nA]', fontsize=text_size, fontweight = 'bold', labelpad=0)
            #ax[k%30].set_ylim([55,120])

        
    np.savetxt(OUTPUT_side+'thresholds'+side+'.txt', thresholds)


 
 
    """
    cbar =fig.colorbar(CS, cax=axC0,ticks=levels,use_gridspec=True)
    cbar.set_label('Axon diameter ['+r'$\mu$'+'m]', rotation=270, fontsize =text_size, fontweight = 'bold',labelpad=24)
    #cbar.set_label('Fibre diameter ['+r'$\mu$'+'m]', rotation=270, fontsize =text_size, fontweight = 'bold',labelpad=24)        
 
    
     
    """
    fig.subplots_adjust(hspace=0.2,wspace=0.2)
 
    plt.show()
