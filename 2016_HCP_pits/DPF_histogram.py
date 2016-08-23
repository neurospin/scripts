"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import nibabel.gifti.giftiio as gio
import os, json, time, glob, argparse
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
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    ax2.spines['left'].set_visible(False)
    for ylabel_i in ax2.get_yticklabels():
        ylabel_i.set_visible(False)
        ylabel_i.set_fontsize(0.0)

    n, bins, patches = ax.hist(X, number_bins, facecolor=COLOR, alpha=alpha, range=(a,b), label=label+": "+str(x.shape[0]), normed=NORMED)
    x_bins=(bins[1:]+bins[:-1])/2
    
    if "DPF" in OUTPUT:
        linspace = np.linspace(-2, 2, 1000).reshape(-1, 1)
    else:
        linspace = np.linspace(int(min(x))+1, int(max(x))+1, 1000).reshape(-1, 1)
    
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
    if abs(mu1)< 200 and  abs(mu2) <200 :
        x_samples = [mu2+(mu1-mu2)/250.0*p for p in range(250)]
        gauss1 = gauss(x_samples, mu1, std1, 1)
        gauss2 = gauss(x_samples, mu2, std2, 1)
        threshold = x_samples[np.argmin(np.abs(gauss1-gauss2))]
    else:
        print "No threshold set"
        #print df
        threshold = 1.0
        arbitrary = True

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
    #ax.set_ylim([0, div*10+10])
    #ax.set_title(r'$\mu$'+'1: '+str(round(mu1,2))+' '+r'$\mu$'+'2: '+str(round(mu2,2))+' '+r'$\mu$'+': '+str(round(mu,2)),fontsize =text_size-4, fontweight = 'bold',verticalalignment="bottom")
    return threshold
    
    #ax.plot(amps,Cvels,color=colorVal,  linestyle='--', marker='o', markersize = 5)
    #ax.set_xscale('log')
 
     
if __name__ == '__main__': 
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('-f', '--feature', type=str,
                        help="Features to measure")
    parser.add_argument('-s', '--symmetric', type=int,
                        help="Boolean, need to give 0 for False and another int for True " 
                        "Specify if symmetric template is used or not")
    parser.add_argument('-d', '--database', type=str,
                        help='Data base from which we take the cluster')
    parser.add_argument('-c', '--collect', type=str,
                        help='Boolean, need to give 0 for False and another int for True '
                        'Specify if re-reading and collecting the distribution is needed')
    options = parser.parse_args()
    feature = options.feature
    SYMMETRIC = bool(int(options.symmetric))
    # The following boolean should be set to true if you want to resample the DPF in each parcel
    # This should be done if you have changed the template parcel for example
    # Else you save ~18min by not re-creating all the files
    COLLECT_PITS_IN_EACH_PARCEL = bool(int(options.collect))
    database_parcel  = options.database
    """
    feature = 'DPF'
    database_parcel = 'hcp'
    SYMMETRIC = True
    COLLECT_PITS_IN_EACH_PARCEL = False

    if SYMMETRIC:
        OUTPUT  = '/neurospin/brainomics/2016_HCP/new_distrib2/distribution_sym_'+feature+'_'+database_parcel+'_Freesurfer_new/'
    else:
        OUTPUT  = '/neurospin/brainomics/2016_HCP/new_distrib2/distribution_'+feature+'_'+database_parcel+'_Freesurfer_new/'
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)

    path0 = '/media/yl247234/SAMSUNG/hcp/Freesurfer_mesh_database/'
    path_parcels = '/media/yl247234/SAMSUNG/'+database_parcel+'/Freesurfer_mesh_database/'
    path = path0+'hcp/'
    temp_file_s_ids='/home/yl247234/s_ids_lists/s_ids.json'
    sides = ['R', 'L']
    with open(temp_file_s_ids, 'r') as f:
        data = json.load(f)
    s_ids  = list(json.loads(data))


    if COLLECT_PITS_IN_EACH_PARCEL:
        for side in sides:
            print "\n"
            print side
            OUTPUT_side = OUTPUT+side+'/'
            if not os.path.exists(OUTPUT_side):
                os.makedirs(OUTPUT_side)

            t0 = time.time()
            if SYMMETRIC:
                file_parcels_on_atlas = path_parcels +'pits_density_update/clusters_total_average_pits_smoothed0.7_60_sym_dist15.0_area100.0_ridge2.0.gii'
            else:
                file_parcels_on_atlas = path_parcels +'pits_density_update/clusters_'+side+'_average_pits_smoothed0.7_60_dist15.0_area100.0_ridge2.0.gii'

            array_parcels = gio.read(file_parcels_on_atlas).darrays[0].data
            parcels_org =  np.unique(array_parcels)
            parcels_org = parcels_org[1:]
            NB_PARCELS = len(parcels_org)
            for k,parcel in enumerate(parcels_org):
                t = time.time()
                X = np.array([])
                for j, s_id in enumerate(s_ids):
                    if SYMMETRIC:
                        file_pits_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                                          "default_analysis", "segmentation", "mesh",
                                                          "surface_analysis_sym", s_id+"_"+side+"white_pits_on_atlas.gii")
                        file_DPF_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                                         "default_analysis", "segmentation",
                                                         #s_id+"_"+side+"white_depth_on_atlas.gii")
                                                         "mesh", "surface_analysis_sym", ""+s_id+"_"+side+"white_"+feature+"_on_atlas.gii")
                    else:
                        file_pits_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                                          "default_analysis", "segmentation", "mesh",
                                                          "surface_analysis", s_id+"_"+side+"white_pits_on_atlas.gii")
                        file_DPF_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                                         "default_analysis", "segmentation",
                                                         #s_id+"_"+side+"white_depth_on_atlas.gii")
                                                         "mesh", "surface_analysis", ""+s_id+"_"+side+"white_"+feature+"_on_atlas.gii")
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
        if SYMMETRIC:
            labels = '/neurospin/brainomics/2016_HCP/LABELS/labelling_sym_template.csv'
            df_labels = pd.read_csv(labels)
            
            file_parcels_on_atlas = path_parcels +'pits_density_update/clusters_total_average_pits_smoothed0.7_60_sym_dist15.0_area100.0_ridge2.0.gii'
        else:
            file_parcels_on_atlas = path_parcels +'pits_density_update/clusters_'+side+'_average_pits_smoothed0.7_60_dist15.0_area100.0_ridge2.0.gii'
        array_parcels = gio.read(file_parcels_on_atlas).darrays[0].data
        parcels_org =  np.unique(array_parcels)
        parcels_org = parcels_org[1:]
        NB_PARCELS = len(parcels_org)
        count  = 0
        thresholds = []
        gs = gridspec.GridSpec(5, 6, width_ratios=[8,8,8,8,8,8], height_ratios=[6,6,6,6,6])
        fig = plt.figure()
        ax = []
        
        
        df_labels.index = df_labels['Name']
        df_labels = df_labels.sort_index()
        df_labels.index = df_labels['Parcel']
        for k,parcel in enumerate(df_labels['Parcel']):
            if count == 30:
                count = 0
                gs = gridspec.GridSpec(5, 6, width_ratios=[8,8,8,8,8,8], height_ratios=[6,6,6,6,6])
                fig = plt.figure()
                ax = []
            ax.append(fig.add_subplot(gs[count]))
            x  = np.loadtxt(OUTPUT_side+'Parcel_'+str(float(parcel))+'.txt')
            thresh = plot_histo(ax[count], x)
            thresholds.append(plot_histo(ax[count], x))

            ax[count].set_title(df_labels.loc[int(parcel)]['Name'],fontsize =text_size-4, fontweight = 'bold',verticalalignment="bottom")

            ax[count].locator_params(axis = 'x', nbins=4)
            ax[count].locator_params(axis='y',nbins=5)
            ax[count].spines['right'].set_visible(False)
            ax[count].spines['top'].set_visible(False)

            ax[count].xaxis.set_ticks_position('none')
            ax[count].yaxis.set_ticks_position('none')
            #ax[count].get_xaxis().tick_bottom()
            #ax[count].get_yaxis().tick_left()


            ax[count].yaxis.set_ticks_position('left')
            if count%6==0:
                ax[count].set_ylabel('Number of pits', fontsize=text_size, fontweight = 'bold', labelpad=0)

            """else:
                ax[count].spines['left'].set_visible(False)
                for ylabel_i in ax[count].get_yticklabels():
                    ylabel_i.set_visible(False)
                    ylabel_i.set_fontsize(0.0)
            """       
            if count < 24:
                ax[count].spines['bottom'].set_visible(False)
                for xlabel_i in ax[count].get_xticklabels():
                    xlabel_i.set_visible(False)
                    xlabel_i.set_fontsize(0.0)
                    #ax[count].set_ylim([60,100])

            else:
                ax[count].xaxis.set_ticks_position('bottom')
                ax[count].set_xlabel('DPF', fontsize=text_size, fontweight = 'bold', labelpad=0)
                #ax[count].set_xlabel('Amplitude stimulus [nA]', fontsize=text_size, fontweight = 'bold', labelpad=0)
                #ax[count].set_ylim([55,120])

            fig.subplots_adjust(hspace=0.2,wspace=0.2)

            count +=1
        np.savetxt(OUTPUT_side+'thresholds'+side+'.txt', thresholds)




        """
        cbar =fig.colorbar(CS, cax=axC0,ticks=levels,use_gridspec=True)
        cbar.set_label('Axon diameter ['+r'$\mu$'+'m]', rotation=270, fontsize =text_size, fontweight = 'bold',labelpad=24)
        #cbar.set_label('Fibre diameter ['+r'$\mu$'+'m]', rotation=270, fontsize =text_size, fontweight = 'bold',labelpad=24)        



        """
        fig.subplots_adjust(hspace=0.2,wspace=0.2)
    #plt.close('all')
    plt.show()
