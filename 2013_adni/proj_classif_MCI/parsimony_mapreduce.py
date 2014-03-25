# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:53:20 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import time
import sys, os, optparse
from multiprocessing import Pool
import numpy as np
import nibabel
from parsimony.estimators import RidgeLogisticRegression_L1_TV
from parsimony.algorithms.explicit import StaticCONESTA
import parsimony.functions.nesterov.tv as tv


def save_model(output_dir, mod, coef, mask=None, **kwargs):
    import os, os.path, pickle, nibabel, numpy
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pickle.dump(mod, open(os.path.join(output_dir, "model.pkl"), "w"))
    for k in kwargs:
        numpy.save(os.path.join(output_dir, k + ".npy"), kwargs[k])
    if mask is not None:
        import nibabel
        mask_data = mask.get_data() != 0
        arr = numpy.zeros(mask_data.shape)
        arr[mask_data] = coef.ravel()
        im_out = nibabel.Nifti1Image(arr, affine=mask.get_affine())
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


def mapper(key):
    output_dir, params = key
    print output_dir, params, XTR.shape, MASK.shape
    

def mapper2(key):
    output_dir, params = key
    alpha, ratio_k, ratio_l, ratio_g = params
    #output_dir = os.path.join(OUTPUT_PATH,
    #             "-".join([str(v) for v in (alpha, ratio_k, ratio_l, ratio_g)]))
    print "START:", output_dir
    #np.asarray([np.sum(YTR == l) for l in np.unique(YTR)]) / float(YTR.size)
    time_curr = time.time()
    beta = None
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g)) # l2, l1, tv penalties
    tv = RidgeLogisticRegression_L1_TV(k, l, g, A, class_weight="auto", output=True,
                               algorithm=StaticCONESTA(max_iter=100))
    tv.fit(XTR, YTR)#, beta)
    y_pred_tv = tv.predict(XTE)
    beta = tv.beta
    #print key, "ite:%i, time:%f" % (len(tv.info["t"]), np.sum(tv.info["t"]))
    print output_dir, "Time ellapsed:", time.time() - time_curr, "ite:%i, time:%f" % (len(tv.info["t"]), np.sum(tv.info["t"]))
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    time_curr = time.time()
    tv.function = tv.A = None # Save disk space
    save_model(output_dir, tv, beta, MASK,
                                  y_pred_tv=y_pred_tv,
                                  y_true=YTE)


def simu_dataset_load():
    from  parsimony import datasets
    n_samples = 500
    shape = (10, 10, 1)
    X3d, y, beta3d, proba = \
        datasets.classification.dice5.load(n_samples=n_samples,
                            shape=shape, snr=5, random_seed=1)
    X = X3d.reshape((n_samples, np.prod(beta3d.shape)))
#    np.save("/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_y.npy", y)
#    np.save("/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_test_idx.npy", np.arange(100, X.shape[0]))
#    np.save("/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_test_mask.npy", np.ones(shape))
#    import nibabel
#    im = nibabel.Nifti1Image(np.ones(shape)), affine=np.eye(4))
#    im.to_filename("/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_test_mask.nii")
    n_train = 100
    XTR = X[:n_train, :]
    YTR = y[:n_train]
    XTE = X[n_train:, :]
    YTE = y[n_train:]
    return XTR, YTR, XTE, YTE


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
    parser.add_option('--mask',
        help='path to mask  file (numpy or niftii)', type=str)
    parser.add_option('--params',
        help='List of parameters quaduplets separated by ";" \
         ex: alpha l2 l1 tv; alpha l2 l1 tv;', type=str)
    parser.add_option('--output',
        help='output directory', type=str)
    parser.add_option('--cores',
        help='Nb cpu cores to use (default %i)' % 8, default=8, type=int)
    parser.add_option('-r', '--reduce',
                          help='Reduce (default %s)' % False, default=False, action='store_true', dest='reduce')
    print sys.argv
    options, args = parser.parse_args(sys.argv)
    #options, args = parser.parse_args(['scripts/2013_adni/proj_classif_MCI/parsimony_mapreduce.py', '--input_x=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_X.npy', '--input_y=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_y.npy', '--test_idx=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_test_idx.npy', '--mask=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_test_mask.npy', '--params=/home/ed203246/data/pylearn-parsimony/params_l2-l1-tv.txt', '--output=/tmp/toto'])
    # DATASET ---------------------------------------------------------------
    if (not options.input_x and options.input_y) or (options.input_x and not options.input_y):
        print "--input_x and --input_y should be both provided or both missing do a simmulation"
        sys.exit(1)
    if not options.input_x:
        print "Simulated data"
        XTR, YTR, XTE, YTE = simu_dataset_load()
    else:
        X = np.load(options.input_x)
        y = np.load(options.input_y)
        if options.test_idx:
            test_idx = np.load(options.test_idx)
            all_mask = np.ones(X.shape[0],dtype=bool)
            all_mask[test_idx] = False
            train_idx = np.where(all_mask)[0]
        else:
            print "all data train==test"
            test_idx = train_idx = np.arange(X.shape[0])
        #print train_idx, test_idx
        XTR = X[train_idx, :]
        XTE = X[test_idx, :]
        YTR = y[train_idx, :]
        YTE = y[test_idx, :]
    print "XTR.shape =", XTR.shape, "; XTE.shape =", XTE.shape

    if not options.mask:
        print "Mask should provided"
    print "MASK:", options.mask
    try:
        MASK = np.load(options.mask)
        A, _ = tv.A_from_mask(MASK)
    except :
        MASK = nibabel.load(options.mask)
        A, _ = tv.A_from_mask(MASK.get_data())

    # PARAMETERS  -------------------------------------------------------------
    if os.path.isfile(options.params):
        f = open(options.params)
        params_list_str = f.readlines()
        f.close()
    else:
        params_list_str = options.params
        params_list_str = params_list_str.split(";")
    params_list_str = [params.strip() for params in params_list_str \
        if len(params.strip())]
    import re
    params_list_str = [re.sub("[ ,]+", "-", params) for params in
        params_list_str]
    print params_list_str
    params_list = [[float(param) for param in params.split("-")] for params in
        params_list_str]
    print "params_list=", zip(params_list_str, params_list)

    # OUTPUT ------------------------------------------------------------------
    if not options.output:
        print "Output directory should be provided"
        sys.exit(1)
    output_dir = options.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    keys = zip([os.path.join(output_dir, o) for o in params_list_str], params_list)
    print keys
    # OTHERS ------------------------------------------------------------------
    nbcores = options.cores
    REDUCE = options.reduce

    # MAP  --------------------------------------------------------------------
    print params_list
    p = Pool(nbcores)
    p.map(mapper, keys)
"""  

python scripts/2013_adni/proj_classif_MCI/parsimony_mapreduce.py \
--input_x=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_X.npy \
--input_y=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_y.npy \
--test_idx=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_test_idx.npy \
--mask=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_test_mask.nii \
--params=$HOME/data/pylearn-parsimony/params_l2-l1-tv.txt --output=/tmp/toto
""" 