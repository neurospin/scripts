# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 13:54:17 2016

@author: ad247405
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from brainomics import plot_utilities
from parsimony.utils import plot_map2d
import parsimony.functions.nesterov.tv
import pca_tv
import sklearn
from sklearn import decomposition
import parsimony.utils.start_vectors as start_vectors
import collections
import argparse
import parsimony.utils as utils

###############################################################################
# Please set Working directory path
###############################################################################
WD = "./"


###############################################################################
# Load dataset
###############################################################################
ftp_url = 'ftp://ftp.cea.fr/pub/unati/brainomics/papers/pcatv/simulations/data/data_100_100_25/%s'


OUTPUT = os.path.join(WD, "simulation_dataset_PCA_results")
if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)



def get_data():
    dataset_filename = 'data.std.npy'
    dataset_url = ftp_url % dataset_filename
    try: # Python 3
        import urllib.request, urllib.parse, urllib.error
        urllib.request.urlretrieve(dataset_url, dataset_filename)
    except ImportError:
        # Python 2
        import urllib
        urllib.urlretrieve(dataset_url, dataset_filename)
    dataset = np.load(os.path.join(WD,  dataset_filename))
    X = dataset
    assert X.shape == (500, 10000)  
    return X



def compute_coefs_from_ratios(global_pen, tv_ratio, l1_ratio):
    ltv = global_pen * tv_ratio
    ll1 = l1_ratio * global_pen * (1 - tv_ratio)
    ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
    assert(np.allclose(ll1 + ll2 + ltv, global_pen))
    return ll1, ll2, ltv
    

###############################################################################
## Models
###############################################################################
MODELS = collections.OrderedDict()
N_COMP = 3
im_shape = (100,100)
Atv = parsimony.functions.nesterov.tv.A_from_shape(im_shape)

# l2, l1, tv penalties
global_pen=0.01
l1_ratio=0.5
tv_ratio=0.5
ll1, ll2, ltv = compute_coefs_from_ratios(global_pen,tv_ratio,l1_ratio)
start_vector = start_vectors.RandomStartVector(seed=24)


MODELS["SparsePCA"] = \
   sklearn.decomposition.SparsePCA(n_components=N_COMP,alpha=1)   


MODELS["ElasticNetPCA"] = \
   pca_tv.PCA_L1_L2_TV(n_components=N_COMP,
                                    l1=ll1, l2=ll2, ltv=1e-6,
                                    Atv=Atv,
                                    criterion="frobenius",
                                    eps=1e-6,
                                    max_iter=100,
                                    inner_max_iter=int(1e4),
                                    output=False,start_vector=start_vector)

MODELS["SPCATV"] = \
    pca_tv.PCA_L1_L2_TV(n_components=N_COMP,
                                    l1=ll1, l2=ll2, ltv=ltv,
                                    Atv=Atv,
                                    criterion="frobenius",
                                    eps=1e-6,
                                    max_iter=100,
                                    inner_max_iter=int(1e4),
                                    output=False,start_vector=start_vector)

###############################################################################
## tests
###############################################################################
def fit_model(model_key):
    global MODELS
    mod = MODELS[model_key]   
    ret = True
    try:       
        mod.fit(X)
        mod.__title__ = "%s\n" % (model_key)
        try:
            mod.__title__ +=\
                "(%i,%i)" % (int(mod.get_info()['converged']),
                             mod.get_info()['num_iter'])
        except:
            pass
    except Exception, e:
        print e
        ret = False
    assert ret

def fit_all(MODELS):
    for model_key in MODELS:
        fit_model(model_key)

def plot_map2d_of_PCA_models(models_dict, nrow, ncol, shape,N_COMP,folder,title_attr=None):
    """Plot 2 weight maps of models"""
    #from .plot import plot_map2d
    ax_i = 1
    for k in models_dict.keys():
        mod = models_dict[k]
        if  hasattr(mod, "V"):
            w = mod.V
        elif hasattr(mod, "components_"): # to work with sklean
            w = mod.components_.T
        if (title_attr is not None and hasattr(mod, title_attr)):
            title = getattr(mod, title_attr)
        else:
            title = None
        ax = plt.subplot(nrow, ncol, ax_i)
        plt.tight_layout()
        utils.plot_map2d(w[:,0].reshape(shape[:2]),ax,title=title)
        ax_i += 1
        ax = plt.subplot(nrow, ncol, ax_i)
        utils.plot_map2d(w[:,1].reshape(shape[:2]),ax,title=title)
        ax_i += 1
        ax = plt.subplot(nrow, ncol, ax_i)
        utils.plot_map2d(w[:,2].reshape(shape[:2]),ax,title=title)
        ax_i += 1
    plt.savefig(os.path.join(folder,'pca_fits.png'))
    plt.close()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', action='store_true', default=False,
                        help="Fit models and plot weight maps")
    parser.add_argument('-m', '--models', help="test only models listed as "
                        "args. Possible models:" + ",".join(MODELS.keys()))
    options = parser.parse_args()
    
    #Load data
    X = get_data()
    #Run PCA
    if options.models:
        import string
        models = string.split(options.models, ",")
        for model_key in MODELS:
            if model_key not in models:
                MODELS.pop(model_key)

    if options.plot:
        fit_all(MODELS)
        plot_map2d_of_PCA_models(MODELS, nrow=3, ncol=3, shape=im_shape,N_COMP=3,
                                 folder = OUTPUT,title_attr="__title__")
                             

                           

# Launch code
#python code_simulated_dataset_ftp.py -p