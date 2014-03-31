# -*- coding: utf-8 -*-
"""
Map reduce for parsimony.
In Mapper mode provide: --input_x, --input_y, --params, --image, --map_output.
It will fork one process per param tuple.
In reducer mode provide: --reduce_input, --reduce_output
"""

import time
import sys, os, glob, optparse, re, fnmatch
from multiprocessing import Pool, Process
import numpy as np
import nibabel
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support

from parsimony.utils import check_arrays
from parsimony.estimators import RidgeLogisticRegression_L1_TV
from parsimony.algorithms.explicit import StaticCONESTA
import parsimony.functions.nesterov.tv as tv

param_sep = " "

def save_model(output_dir, mod, coef, image=None, **kwargs):
    import os, os.path, pickle, nibabel, numpy
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pickle.dump(mod, open(os.path.join(output_dir, "model.pkl"), "w"))
    for k in kwargs:
        numpy.save(os.path.join(output_dir, k + ".npy"), kwargs[k])
    if image is not None:
        if isinstance(image, nibabel.Nifti1Image):
            image_data = image.get_data() != 0
            arr = numpy.zeros(image_data.shape)
            arr[image_data] = coef.ravel()
            im_out = nibabel.Nifti1Image(arr, affine=image.get_affine())
            im_out.to_filename(os.path.join(output_dir, "beta.nii"))
        else:
            image_data = image != 0
            arr = numpy.zeros(image_data.shape)
            arr[image_data] = coef.ravel()
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


def tuples_fromtxt(lines, otype=int):
    """Read resampling text file/lines. Each line is one resample, each resample
    may contains several blocs (train, test) grouped by brackets or semicolons
    item may be separeted by comma or space:
    # first fold:
    [0, 3][1 2]
    # second fold:
    1, 2; 0, 3

    Example:
    -------
    >>> cvfile = "/tmp/cv-train-idx.txt"
    >>> o = open(cvfile, "w")
    >>> for tr, te in [[[0,3],[1,2]], [[1,2],[0,3]]]:
    ...     line = str(tr)+str(te)+"\n"
    ...     o.write(line)
    ... 
    >>> o.close()
    >>> tuples_fromtxt(cvfile)
    [[[0, 3], [1, 2]], [[1, 2], [0, 3]]]
    >>> tuples_fromtxt(lines="[0,3],[1,2]\n[1,2],[0,3]")
    [[[0, 3], [1, 2]], [[1, 2], [0, 3]]]
    >>> tuples_fromtxt(lines="[0,3],[1,2]")
    [[[0, 3], [1, 2]]]
    >>> tuples_fromtxt(lines="0,3 ; 1,2")
    [[[0, 3], [1, 2]]]
    """
    import re
    tuples = list()
    if os.path.isfile(lines):
        i = open(lines)
        lines = i.readlines()
        i.close()
    else:
        lines = lines.splitlines()
    for line in lines:
        #line = lines[0]
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.find(";") is not -1:
            blocs = [bloc.strip() for bloc in line.split(";")]
        else:
            blocs = [bloc.strip() for bloc in re.split('\[([0-9 ,]+)]', line)]
        # rm item without at least number
        blocs = [bloc for bloc in blocs if re.match("[0-9]+", bloc)]
        # split blocs into items
        blocs = [re.split("[ ,]+", bloc) for bloc in blocs] 
        blocs = [[otype(item) for item in bloc] for bloc in blocs]
        if len(blocs) > 0:
            tuples.append(blocs)
    return tuples

def mapper2(key):
    output_dir, params = key
    print output_dir, params, XSR[0][0].shape, XSR[0][1].shape, XSR[1][0].shape, XSR[1][1].shape, IMAGE.shape
    

def mapper(key):
    output_dir, params = key
    alpha, ratio_k, ratio_l, ratio_g = params
    print "START:", output_dir
    time_curr = time.time()
    beta = None
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g)) # l2, l1, tv penalties
    tv = RidgeLogisticRegression_L1_TV(k, l, g, A, class_weight="auto", output=True,
                               algorithm=StaticCONESTA(max_iter=100))
    tv.fit(XsTR[0], XsTR[1])#, beta)
    y_pred_tv = tv.predict(XsTE[0])
    beta = tv.beta
    #print key, "ite:%i, time:%f" % (len(tv.info["t"]), np.sum(tv.info["t"]))
    print output_dir, "Time ellapsed:", time.time() - time_curr, "ite:%i, time:%f" % (len(tv.info["t"]), np.sum(tv.info["t"]))
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    time_curr = time.time()
    tv.function = tv.A = None # Save disk space
    save_model(output_dir, tv, beta, image,
                                  y_pred=y_pred_tv,
                                  y_true=XsTE[1])


if __name__ == "__main__":
    parser = optparse.OptionParser(description=__doc__)
#    parser.add_option('--input_x', '-x', help='X (numpy) dataset path (Required for Mapper)', type=str)
#    parser.add_option('--input_y', '-y', help='y (numpy) dataset path (Required for Mapper)', type=str)
    parser.add_option('--map_cv',
        help='Text file containing cross-validation resamples. '
        'Each line contains one train/test fold. Train/test are grouped '
        'by brackets or semicolons. Items may be separeted by comma or space:'
        '[0 3] [1 2] or [0, 3] [1, 2] or 0, 3; 1, 2. It is also possible to add'
        ' commented lines starting with #',
        type=str)
    parser.add_option('--map_data', help='Path (with wildcard) to numpy datasets (Required for Mapper)', type=str)
    parser.add_option('--map_data_image',
        help='Path to image file (niftii). image IS NOT APPLIED '
        'data are supposed to be imageed, it is only used to generate '
        'niftii output image.', type=str)
    parser.add_option('--map_params',
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
    """
    options, args =  parser.parse_args(['scripts/2013_adni/proj_classif_MCI/parsimony_mapreduce.py', '--map_data=/home/ed203246/data/pylearn-parsimony/datasets/data/classif_10x7x7x1_?.npy', '--map_cv=/home/ed203246/data/pylearn-parsimony/datasets/data/classif_10x7x7x1_cv-train-idx.txt', '--map_data_image=/home/ed203246/data/pylearn-parsimony/datasets/data/classif_10x7x7x1_mask.nii', '--map_params=/home/ed203246/data/pylearn-parsimony/params_alpha-l2-l1-tv_small.txt', '--map_output=/volatile/duchesnay/classif_500x10x10/cv', '--cores=8'])
)
    """
    # =======================================================================
    # == MAP                                                               ==
    # =======================================================================
    # Build Jobs tab    
    # [params, resample, input, output]
    workers = list()
    if options.map_data and options.map_params and options.map_output:
        # -- PARAMETERS  ----------------------------------------------------
        params_list = tuples_fromtxt(options.map_params, otype=str)
        jobs = [param_sep.join(param[0]) for param in params_list]
        ## -- RESAMPLING ----------------------------------------------------
        if options.map_cv:
            cv = tuples_fromtxt(options.map_cv, otype=int)
            jobs = [[job] + [str(fold),
                    os.path.join(options.map_output, str(fold_i), jobs[job_i])]
                for fold_i, fold in enumerate(cv)
                for job_i, job in enumerate(jobs)]
        else:
            jobs = [[job] + [None] for job in jobs]
        if options.map_data_image:
            jobs = [job + [options.map_data_image] for job in jobs]
        else:
            jobs = [job + [None] for job in jobs]
        jobs = [job + [options.map_data] for job in jobs]
        jobs = pd.DataFrame(jobs, columns=["params", "resample", "output", "image", "data"])
        jobs_file = os.path.join(options.map_output, "jobs.csv")
        jobs.to_csv(jobs_file)
        print "Save jobs in", jobs_file
        #sys.exit(0)

        print "** MAP **"
        # Use this to load/slice data only once
        resample_cur = None
        data_cur = None
        image_cur = None
        for i in xrange(jobs.shape[0]):
            #i=0
            job = jobs.iloc[i]
            params = tuples_fromtxt(job["params"], otype=float)[0][0]
            if not data_cur: # Load
                print "Load", job["data"]
                XS = check_arrays(*[np.load(filein) for filein in glob.glob(job["data"])])
                data_cur = job["data"]
            if data_cur != job["data"]:  # re-load data
                print "Load", job["data"]
                XS = check_arrays(*[np.load(filein) for filein in glob.glob(job["data"])])
                data_cur = job["data"]
            if not resample_cur and job["resample"]:  # Load
                resample = tuples_fromtxt(job["resample"], otype=int)[0]
                resample_cur = job["resample"]
            if resample_cur != job["resample"]:
                resample = tuples_fromtxt(job["resample"], otype=int)[0]\
                    if job["resample"] is not None else None
                #XSR X's Resampled look like: [[Xtr, ytr], [Xte, yte]]
            if resample:
                XSR = [[X[idx, :] for X in XS] for idx in resample]
            else: # If not resample create [[Xtr, ytr], [Xte, yte]]
                # where Xtr == Xte and ytr == yte
                XSR = [[X for X in XS] for i in xrange(2)]
            if not image_cur and job["image"]:
                IMAGE = nibabel.load(job["image"])
                A, _ = tv.A_from_mask(IMAGE.get_data())
                image_cur = job["image"]
            if image_cur != job["image"]:  # load image
                IMAGE = nibabel.load(job["image"])
                A, _ = tv.A_from_image(IMAGE.get_data())
                image_cur = job["image"]
            # Job ready to be executed
            key = (job["output"], params)
            # see if we can create a worker
            if len(workers) < options.cores:
                p = Process(target=mapper2, args=(key, ))
                p.start()
                workers.append(p)
            else:
                for p in workers:
                    p.join(1) 
        # -- DATASET --------------------------------------------------------
        
        #X = np.load(options.input_x)
        #y = np.load(options.input_y)
#            test_idx = np.load(options.test_idx)
#            all_image = np.ones(Xs[0].shape[0], dtype=bool)
#            all_image[test_idx] = False
#            train_idx = np.where(all_image)[0]
#            XsTR = [X[train_idx, :] for X in Xs]
#            XsTE = [X[test_idx, :] for X in Xs]
#            #XTE = X[test_idx, :]
#            #YTR = y[train_idx, :]
#            #YTE = y[test_idx, :]
#        else:
#            print "all data train==test"
#            XsTR = XsTE = Xs
#        del(Xs)
#        print "XsTR shapes =", [x.shape for x in XsTR]
#        try:
#            image = np.load(options.map_data_image)
#            A, _ = tv.A_from_image(image)
#        except:
#        print "params_list len =", len(params_list)

        # -- OUTPUT ---------------------------------------------------------
#        output_dir = options.map_output
#        if not os.path.exists(output_dir):
#            os.makedirs(output_dir)
#        keys = zip([os.path.join(output_dir, o) for o in params_list_str],
#                    params_list)
#        nbcores = options.cores
#
#        # MAP  --------------------------------------------------------------
#        print params_list
#        if len(keys) > 1:
#            p = Pool(nbcores)
#            p.map(mapper, keys)
#        else:
#            mapper(keys[0])

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
        #{'cv/0': {'path': 'cv/0/0.010-0.05-0.25-0.70'},
        # 'cv/1': {'path': 'cv/1/0.010-0.05-0.25-0.70'}}
        #print predictions['0.010-0.05-0.25-0.70']['cv/1'].keys()
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
--map_data=/home/ed203246/data/pylearn-parsimony/datasets/data/classif_10x7x7x1_?.npy \
--map_cv=/home/ed203246/data/pylearn-parsimony/datasets/data/classif_10x7x7x1_cv-train-idx.txt \
--map_data_image=/home/ed203246/data/pylearn-parsimony/datasets/data/classif_10x7x7x1_mask.nii \
--map_params=/home/ed203246/data/pylearn-parsimony/params_alpha-l2-l1-tv_small.txt \
--map_output=/volatile/duchesnay/classif_500x10x10/cv \
--cores=2

python scripts/2013_adni/proj_classif_MCI/parsimony_mapreduce.py \
--reduce_input="/volatile/duchesnay/classif_500x10x10/cv" \
--reduce_output="/volatile/duchesnay/classif_500x10x10/scores.csv"
"""

