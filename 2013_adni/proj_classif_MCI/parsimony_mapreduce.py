# -*- coding: utf-8 -*-
"""
Map reduce for parsimony.
"""

import time
import sys, os, glob, argparse, fnmatch
from multiprocessing import Process
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support
from parsimony.utils import check_arrays

param_sep = " "

example = """
Example
-------

# Build a dataset: X, y, mask structure and cv
import numpy as np, nibabel
from  parsimony import datasets
from sklearn.cross_validation import StratifiedKFold
n_samples, shape = 10, (7, 7, 1)
X3d, y, beta3d, proba = datasets.classification.dice5.load(n_samples=n_samples,
shape=shape, snr=5, random_seed=1)
X = X3d.reshape((n_samples, np.prod(beta3d.shape)))
cv = StratifiedKFold(y.ravel(), n_folds=2)

# Save X, y, mask structure and cv
np.save('X.npy', X)
np.save('y.npy', y)
nibabel.Nifti1structure(np.ones(shape), np.eye(4)).to_filename('mask.nii')
# cv file
o = open('cv-train-idx.txt', "w")
for tr, te in cv:
    o.write(str(tr) + str(te) + "\\n")
o.close()

# parameters file
o = open("params_alpha-l2-l1-tv.txt", "w")
params=["0.100 0.25 0.25 0.50", "0.010 0.25 0.25 0.50", "0.001 0.25 0.25 0.50"]
for param in params:
    o.write(param + "\\n")
o.close()

# 1) Build jobs file ---
python parsimony_mapreduce.py --mode build_job \
--data "?.npy" \
--params params_alpha-l2-l1-tv.txt \
--map_output map_results \
--cv cv-train-idx.txt \
--structure mask.nii  \
--job_file jobs.csv

# 2) Map ---
python parsimony_mapreduce.py --mode map \
--job_file jobs.csv \
--user_func user_func_mapreduce.py  \
--cores 2

python parsimony_mapreduce.py \
--reduce_input="/volatile/duchesnay/classif_500x10x10/cv" \
--reduce_output="/volatile/duchesnay/classif_500x10x10/scores.csv"
"""


def jobs_table(options):
# -- PARAMETERS  ----------------------------------------------------
    params_list = tuples_fromtxt(options.params, otype=str)
    jobs = [param_sep.join(param[0]) for param in params_list]
    ## -- RESAMPLING ----------------------------------------------------
    if options.cv:
        cv = tuples_fromtxt(options.cv, otype=int)
        jobs = [[job] + [str(fold),
                os.path.join(options.map_output, str(fold_i), jobs[job_i])]
            for fold_i, fold in enumerate(cv)
            for job_i, job in enumerate(jobs)]
    else:
        jobs = [[job] + [None] for job in jobs]
    if options.structure:
        jobs = [job + [options.structure] for job in jobs]
    else:
        jobs = [job + [None] for job in jobs]
    jobs = [job + [options.data] for job in jobs]
    jobs = pd.DataFrame(jobs, columns=["params", "resample", "output", "structure", "data"])
    return jobs
        #sys.exit(0)

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
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=__doc__, epilog=example)
    parser.add_argument('--mode',
        help='Three possible mode: '
        '(1)"build_job": build jobs csv file. '
           'Required arguments: --data, --params, --map_output, --job_file. '
           'Optional arguments: --cv, --structure. '
        '(2)"map": run jobs file. '
        '(3)"reduce": reduce.')
    # Job builder options ------------------------------------------------------
    # Data, params, output (optional cv and structure)
    parser.add_argument('--data', help='Path (with wildcard) to numpy datasets. '
        'Required option for --mode=build_job.')
    parser.add_argument('--params',
        help='List of parameters tuples (Required for Mapper) separated by ";". '
        'Example: 1.0 0.1 0.1 0.8; 2.0 0.0 0.2 0.8;. '
        'Or a file path containing one line per parameters tuple. '
        'Required option for --mode=build_job.')
    parser.add_argument('--map_output',
        help='Mapper output root directory. '
        'Required option for --mode=build_job.')
    parser.add_argument('--cv',
        help='Text file containing cross-validation resamples. '
        'Each line contains one train/test fold. Train/test are grouped '
        'by brackets or semicolons. Items may be separeted by comma or space:'
        '[0 3] [1 2] or [0, 3] [1, 2] or 0, 3; 1, 2. It is also possible to add'
        ' commented lines starting with #. '
        'Optional option for --mode=build_job.')
    parser.add_argument('--structure',
        help='Path to structure file. This file has the structure of the '
        'original data. it will be provided to user defined '
        'A_from_structure(structure) method. All format are possible.'
        'Optional option for --mode=build_job.')
    parser.add_argument('--job_file',
        help='Path to jobs csv file: used as output in --mode=build_job '
        'and used as input in --mode=map')
    parser.add_argument('--user_func',
        help='User defined functions: (i) A_from_structure(structure_filepath) '
        '(ii) def mapper(key)'
        'Required with --mode=map')
    # Or map_jobs
    parser.add_argument('--cores', '-c',
        help='Nb cpu cores to use (default %i)' % 8, default=8, type=int)

    # Reducer options --------------------------------------------------------
    parser.add_argument('--reduce_input',  help='Reduce input root directory of partial results')
    parser.add_argument('--reduce_output', help='Reduce output, csv file')
    #print sys.argv
    options = parser.parse_args()#sys.argv)
    #print options
    """
    options, args =  parser.parse_args()
    """
    # =======================================================================
    # == BUILD JOBS TABLE                                                  ==
    # =======================================================================
    # ["params", "resample", "output", "structure", "data"]
    if options.mode == "build_job":
        if options.data and options.params and options.map_output \
        and options.job_file:
            jobs = jobs_table(options)
            try:
                os.makedirs(os.path.dirname(options.job_file))
            except:
                pass
            jobs.to_csv(options.job_file)
            print "Save jobs in:", options.job_file
        else:
            print \
            'Required arguments: --data, --params, --map_output, --job_file'
            'Optional arguments: --cv, --structure'
            sys.exit(1)
    # =======================================================================
    # == MAP                                                               ==
    # =======================================================================
    if options.mode == "map":
        if not options.job_file:
            print 'Required arguments: --job_file'
            sys.exit(1)
        if not options.user_func:
            print 'Required arguments: --user_func'
            sys.exit(1)
        exec(open(options.user_func).read() )
        jobs = pd.read_csv(options.job_file)
        print "** MAP WORKERS TO JOBS **"
        # Use this to load/slice data only once
        resample_cur = None
        data_cur = None
        structure_cur = None
        workers = list()
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
                #DATA X's Resampled look like: [[Xtr, ytr], [Xte, yte]]
            if resample:
                DATA = [[X[idx, :] for X in XS] for idx in resample]
            else: # If not resample create [[Xtr, ytr], [Xte, yte]]
                # where Xtr == Xte and ytr == yte
                DATA = [[X for X in XS] for i in xrange(2)]
            if not structure_cur and job["structure"]:
                A, STRUCTURE = A_from_structure(job["structure"])
                structure_cur = job["structure"]
            if structure_cur != job["structure"]:  # load structure
                A, STRUCTURE = A_from_structure(job["structure"])
                structure_cur = job["structure"]
            # Job ready to be executed
            key = (job["output"], params)
            # see if we can create a worker
            #print "len(workers)", len(workers), options.cores
            while len(workers) == options.cores:
                for p in workers:
                    #print "Is alive", str(p), p.is_alive()
                    if not p.is_alive():
                        p.join()
                        workers.remove(p)
                        print "Join :", str(p)
                time.sleep(1)
            p = Process(target=mapper, args=(key, ))
            print "Start:", str(p)
            p.start()
            workers.append(p)

        for p in workers:  # Join remaining worker
            p.join()
            workers.remove(p)
            print "Join :", str(p)


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


