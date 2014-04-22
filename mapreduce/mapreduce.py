#!/usr/bin/env python
"""
Map reduce for parsimony.
"""

import time
import fcntl
import errno
import json
import sys, os, glob, argparse, re
import pickle
import nibabel
from multiprocessing import Process, cpu_count
import numpy as np
import pandas as pd

#from parsimony.utils import check_arrays

DATA = "GLOBAL TOTO"
param_sep = "_"

example = """
Example
-------

## Create dataset
n, p = 10, 5
X = np.random.rand(n, p)
beta = np.random.rand(p, 1)
y = np.dot(X, beta)
np.save('X.npy', X)
np.save('y.npy', y)

## Create config file
cv = [[tr.tolist(), te.tolist()] for tr,te in KFold(n, n_folds=2)]
params = [[1.0, 0.1], [1.0, 0.9], [0.1, 0.1], [0.1, 0.9]]
# enet_userfunc.py is a python file containing mapper(key, output_collector)
# and reducer(key, values) functions.

config = dict(data="?.npy", params=params, user_func="enet_userfunc.py",
              map_output="map_results",
              resample=cv,
              ncore=2,
              reduce_input="map_results/*/*", reduce_group_by=".*/.*/(.*)")
json.dump(config, open("config.json", "w"))

"""


def load_data(key_filename):
    return {key:np.load(key_filename[key]) for key in key_filename}

_table_columns = dict(output=0, params=1, resample_nb=2, data=3)

#RESAMPLE_IN_OPTIONS = "<RESAMPLE IS IN options.resample>"
def _build_job_table(options):
    params_list = json.load(open(options.params)) \
        if isinstance(options.params, str) else options.params
    jobs = [[os.path.join(options.map_output, str(resample_i), 
                          param_sep.join([str(p) for p in params])),
            params,
            resample_i,
            options.data]
            for resample_i in xrange(len(options.resample))
            for params in params_list]
    return jobs

job_template_pbs =\
"""#!/bin/bash
#PBS -S /bin/bash
#PBS -N %(job_name)s
#PBS -l nodes=1:ppn=%(ppn)s
#PBS -l walltime=48:00:00
#PBS -d %(job_dir)s
#PBS -q %(queue)s

%(script)s
"""

def _build_pbs_jobfiles(options):
    cmd_path = os.path.realpath(__file__)
    project_name = os.path.basename(os.path.dirname(options.config))
    job_dir = os.path.dirname(options.config)
    for nb in xrange(options.pbs_njob):
        params = dict()
        params['job_name'] = '%s_%.2i' % (project_name, nb)
        params['ppn'] = options.ncore
        params['job_dir'] = job_dir
        params['script'] = '%s --mode map --config %s' % (cmd_path, options.config)
        params['queue'] = options.pbs_queue
        qsub = job_template_pbs % params
        job_filename = os.path.join(job_dir, 'job_%.2i.pbs' % nb)
        with open(job_filename, 'wb') as f:
            f.write(qsub)
        os.chmod(job_filename, 0777)
    run_all = os.path.join(job_dir, 'jobs_all.sh')
    with open(run_all, 'wb') as f:
        f.write("ls %s/job_*.pbs|while read f ; do qsub $f ; done" % job_dir)
    os.chmod(run_all, 0777)
    sync_push = os.path.join(job_dir, 'sync_push.sh')
    with open(sync_push, 'wb') as f:
        f.write("rsync -azvu %s gabriel.intra.cea.fr:%s/" %
        (os.path.dirname(options.config), os.path.dirname(os.path.dirname(options.config))))
    os.chmod(sync_push, 0777)
    sync_pull = os.path.join(job_dir, 'sync_pull.sh')
    with open(sync_pull, 'wb') as f:
        f.write("rsync -azvu gabriel.intra.cea.fr:%s %s/" %
        (os.path.dirname(options.config), os.path.dirname(os.path.dirname(options.config))))
    os.chmod(sync_pull, 0777)
    print "# 1) Push your file to gabriel, run:"
    print sync_push
    print "# 2) Log on gabriel:"
    print 'ssh -t gabriel.intra.cea.fr "cd %s ; bash"' % os.path.dirname(options.config)
    print "# 3) Run the jobs"
    print run_all
    print "exit"
    print "# 4) Pull your file from gabriel, run"
    print sync_pull

def _makedirs_safe(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def _import_module_from_filename(filename):
    sys.path.append(os.path.dirname(filename))
    name, _ = os.path.splitext(os.path.basename(filename))
    user_module = __import__(os.path.basename(name))
    return user_module

_import_user_func = _import_module_from_filename

class OutputCollector:
    """Map output collector

    Parameters:
    ----------
    output_dir: string output directory, where map results will be stored

    Example:
    --------
    oc = OutputCollector("/tmp/toto")
    oc.is_running()
    oc.set_running(True)
    oc.is_running()
    oc.set_running(False)
    oc.is_running()
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.lock_filename = output_dir + "_lock"
        self.running_filename = output_dir + "_run"
        #if not os.path.exists(os.path.dirname(self.output_dir)):
        _makedirs_safe(os.path.dirname(self.output_dir))

    def lock_acquire(self):
        self.lock_handle = open(self.lock_filename, 'w')
        fcntl.flock(self.lock_handle, fcntl.LOCK_EX)

    def lock_release(self):
        fcntl.flock(self.lock_handle, fcntl.LOCK_UN)
        self.lock_handle.close()
        #os.remove(self.lock_filename)

    def set_running(self, state):
        if state:
            of = open(self.running_filename + "_@%s" % os.uname()[1], 'w')
            of.close()
        else:
            files = glob.glob(self.running_filename + "*")
            if len(files) > 0:
                [os.remove(f) for f in files]
        #print state, self.running_filename, os.path.exists(self.running_filename)

    def is_done(self):
        return os.path.exists(self.output_dir)

    def is_running(self):
        return len(glob.glob(self.running_filename + "*")) > 0

    def collect(self, key, value):
        _makedirs_safe(self.output_dir)
        #if not os.path.exists(self.output_dir):
        #    os.makedirs(self.output_dir)
        for k in value:
            if isinstance(value[k], np.ndarray):
                np.save(os.path.join(self.output_dir, k + ".npy"), value[k])
            elif isinstance(value[k], nibabel.Nifti1Image):
                value[k].to_filename(os.path.join(self.output_dir, k + ".nii"))
            else:
                of = open(os.path.join(self.output_dir, k + ".pkl"), "w")
                pickle.dump(value[k], of)
                of.close()
        self.set_running(False)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.output_dir)

    def load(self):
        res = dict()
        for arr_filename in glob.glob(os.path.join(self.output_dir, "*.npy")):
            #print arr_filename
            name, ext = os.path.splitext(os.path.basename(arr_filename))
            res[name] = np.load(arr_filename)
        for pkl_filename in glob.glob(os.path.join(self.output_dir, "*.pkl")):
            #print pkl_filename
            name, ext = os.path.splitext(os.path.basename(pkl_filename))
            infile = open(pkl_filename, "r")
            res[name] = pickle.load(infile)
            infile.close()
        return res

## Store output_collectors to do some cleaning if killed
output_collectors = list()

def clean_atexit():
    for oc in output_collectors:
        oc.set_running(False)

import atexit
atexit.register(clean_atexit)

if __name__ == "__main__":
    #global DATA
    parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=__doc__, epilog=example)
    parser.add_argument('--mode',
        help='Three possible mode: '
        '(1)"build_job_qsub": build qsub job files '
           'Required arguments: --data, --params, --map_output, --job_file. '
           'Optional arguments: --resample, --structure. '
        '(2)"map": run jobs file. '
        '(3)"reduce": reduce.')

    # Config file
    parser.add_argument('--config', help='Configuration json file that '
        'contains a dictionary of configuration options. There are 4'
        'required argument'
        '(1) "data": (Required) Path (with wildcard) to numpy datasets. '
        'Ex.: "?.npy" will match X.npy and y.npy. '
        '(2) "params":  (Required) List of list of parameters values. Ex:'
        '[[1.0, 0.1], [1.0, 0.9], [0.1, 0.1], [0.1, 0.9]]. '
        '(3) "map_output":  (Required) Mapper output root directory. '
        'Output hierarchy will be '
        'organized as follow: <map_output>/<resample_nb>/<params> and '
        ' <map_output>/0/<params> if no resampling is provided.'
        'override config options.'
        '(4) "user_func": (Required) Path to python file that contains user defined '
        ' functions: (i) A_from_structure(structure_filepath) '
        '(ii) def mapper(key). '
        'There a ?? optional arguments:'
        '"resample": (Optional) List of list of list of indices. '
        'Ex: [[[0, 2], [1, 3]], [[1, 3], [0, 2]]] for cross-validation like '
        'resampling.'
        'or list of list of indices, ex: [[0, 1, 2, 3], [1, 3, 0, 2]]. for '
        'bootstraping/permuation like resampling. '
        '"structure": Path to structure file. This file has the structure of the '
        'original data. it will be provided to user defined '
        'A_from_structure(structure) method. All format are possible. '
        '"ncore", "reduce_input" and "reduce_output": see command line argument.'
        ""
        
        )
    default_nproc = int(cpu_count() / 2)
    parser.add_argument('--ncore',
        help='Nb cpu ncore to use (default %i)' % default_nproc, type=int)

    # Reducer options --------------------------------------------------------
    parser.add_argument('--reduce_input', help='Input root dir for reduce. '
        'Should match  --map_output. Required option for --mode=reduce.')
    parser.add_argument('--reduce_group_by', type=str,
                        help='Regular expression to match the grouping key. Example: MAP_OUTPUT/.*/(.*)  will group by parameters. While (MAP_OUTPUT/.*)/.* will match by resample.')
    parser.add_argument('--reduce_output', help='Reduce output, csv file.')

    # PBD options --------------------------------------------------------
    parser.add_argument('--pbs_njob', type=int,
                        help='Build n PBS job file in the '
                             'same directory than the config file')
    default_pbs_queue = "Cati_LowPrio"
    parser.add_argument('--pbs_queue',
                        help='PBS queue (default %s)' % default_pbs_queue)

    #print sys.argv
    options = parser.parse_args()

    if not options.config:
        print 'Required arguments --config'
        sys.exit(1)
    config = json.load(open(options.config))
    for k in config:
        if not hasattr(options, k) or getattr(options, k) is None:
            setattr(options, k, config[k])

    if not hasattr(options, "resample"):
        setattr(options, "resample", None)
    if not hasattr(options, "user_func"):
        setattr(options, "user_func", None)
    if not hasattr(options, "job_file"):
        setattr(options, "job_file", None)
    if options.ncore is None:
        options.ncore = default_nproc
    if options.pbs_queue is None:
        options.pbs_queue = default_pbs_queue

    # import itself to modify global variables (glob.DATA)
    sys.path.append(os.path.dirname(__file__))
    import mapreduce as GLOBAL  # to access to global variables
#    glob.DATA = 33
#    user_func = _import_user_func(options.user_func)
#    user_func.test()
#    print "TOTO"
#    sys.exit(1)

    # =======================================================================
    # == BUILD JOBS TABLE                                                  ==
    # =======================================================================
    # ["params", "resample", "output", "structure", "data"]
    if options.pbs_njob:
        _build_pbs_jobfiles(options)

    # =======================================================================
    # == MAP                                                               ==
    # =======================================================================
    if options.mode == "map":
        if not options.user_func:
            print 'Required fields in config file: "user_func"'
            sys.exit(1)
        user_func = _import_user_func(options.user_func)
        #exec(open(options.user_func).read())
        if hasattr(options, "structure"):
            GLOBAL.A, GLOBAL.STRUCTURE = user_func.A_from_structure(options.structure)
        #jobs = pd.read_csv(options.job_file)
        if options.job_file:
            jobs = json.load(open(options.job_file))
        else:
            jobs = _build_job_table(options)
        print "** MAP WORKERS TO JOBS **"
        # Use this to load/slice data only once
        resamples_file_cur = resample_nb_cur = None
        data_cur = None
        workers = list()
        T = _table_columns
        for i in xrange(len(jobs)):
            # see if we can create a worker
            while len(workers) == options.ncore:
                for p in workers:
                    #print "Is alive", str(p), p.is_alive()
                    if not p.is_alive():
                        p.join()
                        workers.remove(p)
                        print "Joined:", str(p)
                time.sleep(1)
            job = jobs[i]
            output_collector = OutputCollector(job[T["output"]])
            output_collector.lock_acquire()
            if output_collector.is_done() or output_collector.is_running():
                output_collector.lock_release()
                continue
            else:
                output_collectors.append(output_collector)
                output_collector.set_running(True)
                output_collector.lock_release()
            if not data_cur or (data_cur != job[T["data"]]):  # re-load data
                print "Load", job[T["data"]]
                dat_orig = load_data(job[T["data"]])
                data_cur = job[T["data"]]
            if (not resample_nb_cur and job[T["resample_nb"]]) or \
               (resample_nb_cur != job[T["resample_nb"]]):  # Load
                resample_nb_cur = job[T["resample_nb"]]
                resample = options.resample[resample_nb_cur]
                # Except a list of index, if not
                try:                
                    resample[0][0]
                except TypeError:
                    resample = [resample]
                #DATA X's Resampled look like: [[Xtr, ytr], [Xte, yte]]
            if resample:
                GLOBAL.DATA = {k:[dat_orig[k][idx, :]  for idx in resample] for k in dat_orig}
            else: # If not resample create {X:[Xtr, ytr], y:[Xte, yte]}
                # where Xtr == Xte and ytr == yte
                GLOBAL.DATA = {k:[dat_orig[k]  for i in xrange(2)] for k in dat_orig}
            # Job ready to be executed
            #key = (job[P["output"]], params)
            key = job[T["params"]]
            p = Process(target=user_func.mapper, args=(key, output_collector))
            print "Start :", str(p), str(output_collector)
            p.start()
            workers.append(p)

        for p in workers:  # Join remaining worker
            p.join()
            workers.remove(p)
            print "Joined:", str(p)


    # =======================================================================
    # == REDUCE                                                            ==
    # =======================================================================
    if options.mode == "reduce":
        if not options.reduce_input:
            print 'Required arguments: --reduce_input'
            sys.exit(1)
        if not options.user_func:
            print 'Required arguments: --user_func'
            sys.exit(1)
        user_func = _import_user_func(options.user_func)
        print "** REDUCE **"
        items = glob.glob(options.reduce_input)
        items = [item for item in items if os.path.isdir(item)]
        #print items
        options.reduce_group_by
        group_keys = set([re.findall(options.reduce_group_by, item)[0] for item
            in items])
        #print group_keys
        #print items
        groups = {k:[] for k in group_keys}
#        groups['0.010 0.25 0.25 0.50']
#        [{'y_true': [], 'model':, 'y_pred': []},
#         {'y_true': [], 'model':, 'y_pred': []}]
        #print groups
        #print options.reduce_group_by
        for item in items:
            #print item
            which_group_key = [k for k in groups if re.findall(options.reduce_group_by, item)[0]==k]
            #print which_group_key
            if len(which_group_key) != 1:
                raise ValueError("Many/No keys match %s" % item)
            output_collector = OutputCollector(item)
            print "load", output_collector
            groups[which_group_key[0]].append(output_collector.load())
        #print groups
        # Do the reduce
        scores = [user_func.reducer(key=k, values=groups[k]) for k in groups]
        scores = pd.DataFrame(scores)
        print scores.to_string()
        if options.reduce_output is not None:
            scores.to_csv(options.reduce_output, index=False)


