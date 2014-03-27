# -*- coding: utf-8 -*-
"""
Map reduce for parsimony.
In Mapper mode provide: --input_x, --input_y, --params, --mask, --map_output.
It will fork one process per param tuple.
In reducer mode provide: --reduce_input, --reduce_output
"""

import time
import sys, os, optparse, re, fnmatch
from multiprocessing import Pool
import numpy as np
import nibabel
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support

from parsimony.estimators import RidgeLogisticRegression_L1_TV
from parsimony.algorithms.explicit import StaticCONESTA
import parsimony.functions.nesterov.tv as tv

param_sep = "-"

def save_model(output_dir, mod, coef, mask=None, **kwargs):
    import os, os.path, pickle, nibabel, numpy
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pickle.dump(mod, open(os.path.join(output_dir, "model.pkl"), "w"))
    for k in kwargs:
        numpy.save(os.path.join(output_dir, k + ".npy"), kwargs[k])
    if mask is not None:
        if isinstance(mask, nibabel.Nifti1Image):
            mask_data = mask.get_data() != 0
            arr = numpy.zeros(mask_data.shape)
            arr[mask_data] = coef.ravel()
            im_out = nibabel.Nifti1Image(arr, affine=mask.get_affine())
            im_out.to_filename(os.path.join(output_dir, "beta.nii"))
        else:
            mask_data = mask != 0
            arr = numpy.zeros(mask_data.shape)
            arr[mask_data] = coef.ravel()
            np.save(os.path.join(output_dir, "beta.nii"), arr)

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


def mapper2(key):
    output_dir, params = key
    print output_dir, params, XTR.shape, MASK.shape
    

def mapper(key):
    output_dir, params = key
    alpha, ratio_k, ratio_l, ratio_g = params
    print "START:", output_dir
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
                                  y_pred=y_pred_tv,
                                  y_true=YTE)


if __name__ == "__main__":
    parser = optparse.OptionParser(description=__doc__)
    parser.add_option('--input_x', '-x', help='X (numpy) dataset path (Required for Mapper)', type=str)
    parser.add_option('--input_y', '-y', help='y (numpy) dataset path (Required for Mapper)', type=str)
    parser.add_option('--test_idx',  '-i',
        help='Numpy file containing test index subjects, if missing train == test', type=str)
    parser.add_option('--mask', '-m',
        help='path to mask  file (numpy or niftii) (Required for Mapper)', type=str)
    parser.add_option('--params', '-p',
        help='List of parameters tuples (Required for Mapper) separated by ";". ' +\
         'Example: 1.0 0.1 0.1 0.8; 2.0 0.0 0.2 0.8;. ' +\
         'Or a file path containing one line per parameters tuple.', type=str)
    parser.add_option('--map_output', '-o',
        help='Mapper output root directory  (Required for Reducer)', type=str)
    parser.add_option('--cores', '-c',
        help='Nb cpu cores to use (default %i)' % 8, default=8, type=int)
    parser.add_option('--reduce_input',  help='Reduce input root directory of partial results', type=str)
    parser.add_option('--reduce_output', help='Reduce output, csv file', type=str)
    print sys.argv
    options, args = parser.parse_args(sys.argv)
    #options, args = parser.parse_args(['scripts/2013_adni/proj_classif_MCI/parsimony_mapreduce.py', '--input_x=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x500x500_X.npy', '--input_y=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x500x500_y.npy', '--test_idx=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x500x500_test_idx.npy', '--mask=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x500x500_test_mask.npy', '--params=/home/ed203246/data/pylearn-parsimony/params_alpha-l2-l1-tv.txt', '--output=/tmp/toto'])
    # =======================================================================
    # == MAP                                                               ==
    # =======================================================================
    if options.input_x and options.input_y and options.mask and \
        options.params and options.map_output:
        print "** MAP **"
        # -- DATASET --------------------------------------------------------
        X = np.load(options.input_x)
        y = np.load(options.input_y)
        if options.test_idx:
            test_idx = np.load(options.test_idx)
            all_mask = np.ones(X.shape[0], dtype=bool)
            all_mask[test_idx] = False
            train_idx = np.where(all_mask)[0]
            XTR = X[train_idx, :]
            XTE = X[test_idx, :]
            YTR = y[train_idx, :]
            YTE = y[test_idx, :]
        else:
            print "all data train==test"
            XTR = XTE = X
            YTR = YTE = y
        del(X)
        del(y)
        print "XTR.shape =", XTR.shape, "; XTE.shape =", XTE.shape
        try:
            MASK = np.load(options.mask)
            A, _ = tv.A_from_mask(MASK)
        except:
            MASK = nibabel.load(options.mask)
            A, _ = tv.A_from_mask(MASK.get_data())

        # -- PARAMETERS  ----------------------------------------------------
        if os.path.isfile(options.params):
            f = open(options.params)
            params_list_str = f.readlines()
            f.close()
        else:
            params_list_str = options.params
            params_list_str = params_list_str.split(";")
        # remove line starting with # strip etc.
        params_list_str = [l.strip() for l in params_list_str]
        params_list_str = [re.sub("#.*", "", l) for l in params_list_str]
        params_list_str = [l.strip() for l in params_list_str]
        params_list_str = [l for l in params_list_str if len(l)]
        params_list_str = [re.sub("[ ,]+", param_sep, l) for l in params_list_str]
        params_list = [[float(param) for param in params.split(param_sep)] for params in
            params_list_str]
        print "params_list len =", len(params_list)

        # -- OUTPUT ---------------------------------------------------------
        output_dir = options.map_output
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        keys = zip([os.path.join(output_dir, o) for o in params_list_str],
                    params_list)
        nbcores = options.cores

        # MAP  --------------------------------------------------------------
        print params_list
        if len(keys) > 1:
            p = Pool(nbcores)
            p.map(mapper, keys)
        else:
            mapper(keys[0])

    # =======================================================================
    # == REDUCE                                                            ==
    # =======================================================================
    if options.reduce_input is not None:
        print "** REDUCE **"
        #print options.reduce_input 
        #reduce_input = "/volatile/duchesnay/classif_500x500x500/cv"
        #reduce_output = "/volatile/duchesnay/classif_500x500x500/scores.csv"
        items = []
        for dirpath, dirnames, filenames in os.walk(options.reduce_input):
          for filename in fnmatch.filter(filenames, 'model.pkl'):
              items.append(dirpath)
        #predictions = by_param dict of  by_groups dict
        #print predictions['0.010-0.05-0.25-0.70']
        #{'/volatile/duchesnay/classif_500x500x500/cv/0': {'path': '/volatile/duchesnay/classif_500x500x500/cv/0/0.010-0.05-0.25-0.70'},
        # '/volatile/duchesnay/classif_500x500x500/cv/1': {'path': '/volatile/duchesnay/classif_500x500x500/cv/1/0.010-0.05-0.25-0.70'}}
        #print predictions['0.010-0.05-0.25-0.70']['/volatile/duchesnay/classif_500x500x500/cv/1'].keys()
        #['path', 'y_true', 'model', 'y_pred']
        predictions = dict()
        for item in items:
            key_arg = os.path.basename(item)
            key_group = os.path.dirname(item)
            if not key_arg in predictions:
                predictions[key_arg] = dict()
            #if not key_group in predictions[key_arg]: predictions[key_arg][key_group] = dict(dir_path=item)
            print "load", item
            item_res = load_model(item)
            item_res["path"] = item
            predictions[key_arg][key_group] = item_res
        scores = list()
        for key_param in predictions:
            cur_param = predictions[key_param]
            y_true = list()
            y_pred = list()
            for key_group  in cur_param:
                y_true.append(cur_param[key_group]["y_true"].ravel())
                y_pred.append(cur_param[key_group]["y_pred"].ravel())
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                         average=None)
            scores.append(key_param.split(param_sep) +
            p.tolist() + [p.mean()] +
            r.tolist() + [r.mean()] +
            f.tolist() + [f.mean()] +
            s.tolist())
        scores = pd.DataFrame(scores,
            columns=["alpha", "l2_ratio", "l1_ratio", "tv_ratio",
                     "recall_0", "recall_1", "recall_mean",
                     "precision_0", "precision_1", "precision_mean",
                     "f1_0", "f1_1", "f1_mean",
                     "support_0", "support_1"])
        scores = scores.sort("recall_mean", ascending=False)
        print scores.to_string()
        if options.reduce_output is not None:
            scores.to_csv(options.reduce_output, index=False)

"""
python scripts/2013_adni/proj_classif_MCI/parsimony_mapreduce.py \
--input_x=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x500x500_X.npy \
--input_y=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x500x500_y.npy \
--test_idx=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x500x500_test_idx.npy \
--mask=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x500x500_test_mask.nii \
--params=/home/ed203246/data/pylearn-parsimony/params_alpha-l2-l1-tv.txt \
--map_output=/volatile/duchesnay/classif_500x500x500/cv/0 \
--cores=8

python scripts/2013_adni/proj_classif_MCI/parsimony_mapreduce.py \
--reduce_input="/volatile/duchesnay/classif_500x500x500/cv" \
--reduce_output="/volatile/duchesnay/classif_500x500x500/scores.csv"

python scripts/2013_adni/proj_classif_MCI/parsimony_mapreduce.py.new.py \
--input_x=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_X.npy \
--input_y=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_y.npy \
--test_idx=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_test_idx.npy \
--mask=/home/ed203246/data/pylearn-parsimony/datasets/classif_500x10x10_test_mask.nii \
--params=/home/ed203246/data/pylearn-parsimony/params_alpha-l2-l1-tv.txt \
--map_output=/volatile/duchesnay/classif_500x10x10/cv/0 \
--cores=8

python scripts/2013_adni/proj_classif_MCI/parsimony_mapreduce.py \
--reduce_input="/volatile/duchesnay/classif_500x10x10/cv" \
--reduce_output="/volatile/duchesnay/classif_500x10x10/scores.csv"
"""

