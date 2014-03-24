# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:53:20 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import time
import sys, optparse
import numpy as np
from parsimony.estimators import RidgeLogisticRegression_L1_TV
from parsimony.algorithms.explicit import StaticCONESTA


def save_model(output_dir, mod, coef, mask_im=None, **kwargs):
    import os, os.path, pickle, nibabel, numpy
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pickle.dump(mod, open(os.path.join(output_dir, "model.pkl"), "w"))
    for k in kwargs:
        numpy.save(os.path.join(output_dir, k + ".npy"), kwargs[k])
    if mask_im is not None:
        mask = mask_im.get_data() != 0
        arr = numpy.zeros(mask.shape)
        arr[mask] = coef.ravel()
        im_out = nibabel.Nifti1Image(arr, affine=mask_im.get_affine())
        im_out.to_filename(os.path.join(output_dir, "beta.nii"))


def load_model(input_dir):
    #input_dir = '/neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv/10-0.1-0.4-0.5'
    import os, os.path, pickle, numpy, glob
    res = dict()
    for arr_filename in glob.glob(os.path.join(input_dir, "*.npy")):
        #print arr_filename
        name, ext = os.path.splitext(os.path.basename(arr_filename))
        res[name] = numpy.load(arr_filename)
    for pkl_filename in glob.glob(os.path.join(input_dir, "*.pkl")):
        #print pkl_filename
        name, ext = os.path.splitext(os.path.basename(pkl_filename))
        res[name] = pickle.load(open(pkl_filename, "r"))
    return res


def mapper(output_dir, params_list):
    alpha, ratio_k, ratio_l, ratio_g = params
    output_dir =  os.path.join(OUTPUT_PATH, "%.2f-%.3f-%.3f-%.2f" % (alpha, ratio_k, ratio_l, ratio_g))
    #output_dir = os.path.join(OUTPUT_PATH,
    #             "-".join([str(v) for v in (alpha, ratio_k, ratio_l, ratio_g)]))
    print "START:", output_dir
    #np.asarray([np.sum(ytr == l) for l in np.unique(ytr)]) / float(ytr.size)
    time_curr = time.time()
    beta = None
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g)) # l2, l1, tv penalties
    tv = RidgeLogisticRegression_L1_TV(alpha, 0, 0, A, algorithm=StaticCONESTA(max_iter=100))
    tv = RidgeLogisticRegression_L1_TV(k, l, g, A, class_weight="auto", output=True,
                               algorithm=StaticCONESTA(max_iter=100))
    tv.fit(Xtr, ytr)#, beta)
    y_pred_tv = tv.predict(Xte)
    beta = tv.beta
    #print key, "ite:%i, time:%f" % (len(tv.info["t"]), np.sum(tv.info["t"]))
    print output_dir, "Time ellapsed:", time.time() - time_curr, "ite:%i, time:%f" % (len(tv.info["t"]), np.sum(tv.info["t"]))
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    time_curr = time.time()
    tv.function = tv.A = None # Save disk space
    save_model(output_dir, tv, beta, mask_im,
                                  y_pred_tv=y_pred_tv,
                                  y_true=yte)


def simu_dataset_load():
    from  parsimony import datasets
    n_samples = 500
    shape = (10, 10, 1)
    X3d, y, beta3d, proba = \
        datasets.classification.dice5.load(n_samples=n_samples,
                            shape=shape, snr=5, random_seed=1)
    X = X3d.reshape((n_samples, np.prod(beta3d.shape)))
    #A, n_compacts = tv.A_from_shape(beta3d.shape)
    #plt.plot(proba[y.ravel() == 1], "ro", proba[y.ravel() == 0], "bo")
    #plt.show()
    n_train = 100
    Xtr = X[:n_train, :]
    ytr = y[:n_train]
    Xte = X[n_train:, :]
    yte = y[n_train:]
    return Xtr, ytr, Xte, yte


if __name__ == "__main__":
    parser = optparse.OptionParser(description=__doc__)
    parser.add_option('--input_x',
        help='X dataset path, if missing simulate data (default %s)' % None,
        default=None, type=str)
    parser.add_option('--input_y',
        help='y dataset path, if missing simulate data (default %s)' % None,
        default=None, type=str)
    parser.add_option('--test_idx',
        help='numpy file containing test index subjects, if missing train == test', type=str)
    parser.add_option('--params',
        help='List of parameters quaduplets separated by ";" \
         ex: alpha l2 l1 tv; alpha l2 l1 tv;', type=str)
    parser.add_option('--output',
        help='output directory', type=str)
    parser.add_option('--cores',
        help='Nb cpu cores to use (default %i)' % 8, default=8, type=int)
    parser.add_option('-r', '--reduce',
                          help='Reduce (default %s)' % False, default=False, action='store_true', dest='reduce')
    
    options, args = parser.parse_args(sys.argv)
    #options, args = parser.parse_args(['../../2013_adni/proj_classif_MCI/parsimony_mapreduce.py'])

    # DATASET ---------------------------------------------------------------
    INPUT_X = options.input_x
    INPUT_y = options.input_y
    if (not INPUT_X and INPUT_y) or (INPUT_X and not INPUT_y):
        print "input_x and input_y sould be both provided or both missing (simmulation)"
        sys.exit(1)
    if not INPUT_X:
        print "Simulated data"
        Xtr, ytr, Xte, yte = simu_dataset_load()
        #X = np.r_[Xtr, Xte]; np.save("/tmp/X.npy", X)
        #y = np.r_[ytr, yte]; np.save("/tmp/y.npy", y)
        #test_idx = np.arange(Xtr.shape[0], Xtr.shape[0]+Xte.shape[0]); np.save("/tmp/test_idx.npy", test_idx)
    else:
        X = np.load(INPUT_X)
        y = np.load(INPUT_y)
        if options.test_idx:
            test_idx = np.load(options.test_idx)
            all_mask = np.ones(X.shape[0],dtype=bool)
            all_mask[test_idx] = False
            train_idx = np.where(all_mask)[0]
        else:
            print "all data train==test"
            test_idx = train_idx = np.arange(X.shape[0])
        #print train_idx, test_idx
        Xtr = X[train_idx, :]
        Xte = X[test_idx, :]
        ytr = y[train_idx, :]
        yte = y[test_idx, :]
    
    print "Xtr.shape =", Xtr.shape, "; Xte.shape =", Xte.shape

    # PARAMETERS  -----------------------------------------------------------
    PARAMS_LIST_STR = options.params
    PARAMS_LIST_STR=" 1,  0.1  0.1 0.8; 0.1 0.1 0.1 0.8;;"
    PARAMS_LIST_STR = PARAMS_LIST_STR.split(";")
    PARAMS_LIST_STR = [params for params in PARAMS_LIST_STR if len(params)]
    import re
    PARAMS_LIST_STR = [re.sub("[ ,]+", "-", params.strip()) for params in PARAMS_LIST_STR]
    PARAMS_LIST = [[float(param) for param in params.split("-")] for params in 
        PARAMS_LIST_STR]
    print "PARAMS_LIST=", PARAMS_LIST

    # OUTPUT ----------------------------------------------------------------
    if not options.output:
        print "Output directory should be provided"
        sys.exit(1)
    OUTPUT = options.output
    PARAMS_LIST =  [[1.0, 0.1, 0.1, 0.853], [0.1, 0.1, 0.1, 0.8], [0.01, 0.1, 0.1, 0.8]]
    precision  = [0 for i in xrange(len(PARAMS_LIST[0]))]
    for params in PARAMS_LIST:
        for i, p in enumerate(params):
            k = 0
            while np.round(p, k) != p:
                k += 1
            precision[i] = k if k > precision[i] else precision[i]
    import string
    output_formating_string = string.join(['%.'+'%if' % p for p in precision], sep="-")
    
    
    # OTHERS ----------------------------------------------------------------
    NBCORES = options.cores    
    REDUCE = options.reduce

"""  
python ../../2013_adni/proj_classif_MCI/parsimony_mapreduce.py --input_x=/tmp/X.npy --input_y=/tmp/y.npy  --test_idx=/tmp/test_idx.npy --params="1 .1 .1 .8; .1 .1 .1 .8"
""" 