#!/usr/bin/env python
"""
Map reduce for parsimony.
"""

import time
import errno
import json
import sys, os, glob, argparse
#import warnings
import pickle
import nibabel
from multiprocessing import Process, cpu_count
import numpy as np
import pandas as pd
from collections import OrderedDict

# Global data
DATA = dict()
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

config = dict(data=dict(X="X.npy", y="y.npy"),
              params=params,
              user_func="enet_userfunc.py",
              map_output="map_results",
              resample=cv,
              ncore=2,
              reduce_group_by="params_str")
json.dump(config, open("config.json", "w"))

"""


def load_data(key_filename):
    return {key: np.load(key_filename[key]) for key in key_filename}

_RESAMPLE_INDEX = 'resample_index'
_PARAMS = 'params'
_PARAMS_STR = 'params_str'
_OUTPUT = 'output dir'
_OUTPUT_COLLECTOR = 'output collector'

GROUP_BY_VALUES = [_RESAMPLE_INDEX, _PARAMS]
DEFAULT_GROUP_BY = _PARAMS


def _build_job_table(config):
    """Build a pandas dataframe representing the jobs.
    The dataframe has 3 columns whose name is given by global variables:
      - _RESAMPLE_INDEX: the index of the resampling
      - _PARAMS: the key passed to map (tuple of parameters)
      - _PARAMS_STR: representation of the parameters as a string (used in
         output dir)
      - _OUTPUT: the output directory
      - _OUTPUT_COLLECTOR: the OutputCollector
    In order to be able to group by parameters, they must be hashable (it's the
    case for tuples made of strings and floats).
    Note that the index respects the natural ordering of (resample, params) as
    given in the config file.
    """
    params_list = json.load(open(config["params"])) \
        if isinstance(config["params"], str) else config["params"]
    # Create representation of parameters as a string
    params_str_list = [param_sep.join([str(p) for p in params])
                       for params in params_list]
    # The parameters are given as list of values.
    # As list are not hashable, we cast them to tuples.
    jobs = [[resample_i,
             tuple(params),
             params_str,
             os.path.join(config["map_output"],
                          str(resample_i),
                          params_str)]
            for resample_i in xrange(len(config["resample"]))
            for (params, params_str) in zip(params_list, params_str_list)]
    jobs = pd.DataFrame.from_records(jobs,
                                     columns=[_RESAMPLE_INDEX,
                                              _PARAMS,
                                              _PARAMS_STR,
                                              _OUTPUT])
    # Add OutputCollectors (we need all the rows so we do that latter)
    jobs[_OUTPUT_COLLECTOR] = jobs[_OUTPUT].map(lambda x: OutputCollector(x))
    return jobs


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

    def clean(self):
        if os.path.exists(self.output_dir) \
            and (len(os.listdir(self.output_dir)) == 0):
            print "clean", self.output_dir
            os.rmdir(self.output_dir)

    def collect(self, key, value):
        _makedirs_safe(self.output_dir)
        for k in value:
            if isinstance(value[k], np.ndarray):
                np.savez_compressed(os.path.join(self.output_dir, k), value[k])
            elif isinstance(value[k], nibabel.Nifti1Image):
                value[k].to_filename(os.path.join(self.output_dir, k + ".nii"))
            else:
                try:
                    of = open(os.path.join(self.output_dir, k + ".json"), "w")
                    json.dump(value[k], of)
                    of.close()
                except:
                    of = open(os.path.join(self.output_dir, k + ".pkl"), "w")
                    pickle.dump(value[k], of)
                    of.close()

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.output_dir)

    def load(self, pattern="*"):
        #print os.path.join(self.output_dir, pattern)
        res = dict()
        for filename in glob.glob(os.path.join(self.output_dir, pattern)):
            o = None
            try:
                fd = np.load(filename)
                if hasattr(fd, "keys"):
                    o = fd['arr_0']
                    fd.close()
                else:
                    o = fd
            except:
                try:
                    fd = open(filename, "r")
                    o = json.load(fd)
                    fd.close()
                except:
                    try:
                        fd = open(filename, "r")
                        o = pickle.load(fd)
                        fd.close()
                    except:
                        try:
                            o = nibabel.load(filename)
                        except:
                            pass
            if o is not None:
                name, ext = os.path.splitext(os.path.basename(filename))
                res[name] = o
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__, epilog=example)

    parser.add_argument('-m', '--map', action='store_true', default=False,
                        help="Run mapper: iterate over resamples and "
                        "parameters, and call mapper (defined in user_func)")

    parser.add_argument('-c', '--clean', action='store_true', default=False,
                        help="Clean execution: remove empty directories "
                        "in map_output. Use it if you suspect that some "
                        "mapper jobs did not end not properly.")

    parser.add_argument('-r', '--reduce', action='store_true', default=False,
                        help="Run reducer: iterate over map_output and call"
                        "reduce (defined in user_func)")

    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help="Force call to mapper even if output is present")

    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    # Config file
    parser.add_argument('config', help='Configuration json file that '
        'contains a dictionary of configuration options. There are 4 '
        'required arguments:'
        '(1) "data": (Required) dict(X="/tmp/X.npy", y="/tmp/X.npy").'
        '(2) "params":  (Required) List of list of parameters values. Ex:'
        '[[1.0, 0.1], [1.0, 0.9], [0.1, 0.1], [0.1, 0.9]]. '
        '(3) "map_output":  (Required) Mapper output root directory. '
        'Output hierarchy will be '
        'organized as follow: <map_output>/<resample_nb>/<params> and '
        ' <map_output>/0/<params> if no resampling is provided.'
        'override config options.'
        '(4) "user_func": (Required) Path to python file that contains 4 user '
        'defined functions: '
        '(i) load_globals(config): executed once at the beginning to load '
        'all the data '
        '(ii) resample() is executed on each new resampling'
        '(iii) mapper(key) is executed on each parameters x reample item'
        '(iv) reducer(key, values)'
        '"resample": (Optional) List of list of list of indices. '
        'Ex: [[[0, 2], [1, 3]], [[1, 3], [0, 2]]] for cross-validation like '
        'resampling.'
        'or list of list of indices, ex: [[0, 1, 2, 3], [1, 3, 0, 2]]. for '
        'bootstraping/permuation like resampling. '
        '"ncore", and "reduce_output": see command line argument.'
        )

    default_nproc = cpu_count()
    parser.add_argument('--ncore',
        help='Nb of cpu cores to use (default %i)' % default_nproc, type=int)
    options = parser.parse_args()

    if not options.config:
        print >> sys.stderr, 'Required config file'
        sys.exit(os.EX_USAGE)
    config_filename = options.config

    # Check that at least one mode is given
    if not (options.map or options.reduce or options.clean):
        print >> sys.stderr, 'Require a mode (map, reduce or clean)'
        sys.exit(os.EX_USAGE)
    # Check that only one mode is given
    if (options.map and options.reduce) or \
       (options.map and options.clean) or \
       (options.reduce and options.clean):
           print >> sys.stderr, 'Require only one mode (map, reduce or clean)'
           sys.exit(os.EX_USAGE)

    ## TO DEBUG just set config_filename here
    # config_filename = "/neurospin/brainomics/2013_adni/proj_classif_MCIc-MCInc_gtvenet/config.json"
    # Read config file
    config_filename = os.path.abspath(config_filename)
    config = json.load(open(config_filename))
    # Set WD to be the dir on config file, this way all path can be relative
    os.chdir(os.path.dirname(config_filename))

    if options.ncore is None:
        options.ncore = default_nproc

    #=========================================================================
    #== Check config file                                                   ==
    #=========================================================================
    if "user_func" not in config:
        print >> sys.stderr, 'Attribute "user_func" is required'
        sys.exit(os.EX_CONFIG)
    user_func = _import_user_func(config["user_func"])

    if "reduce_group_by" not in config:
        config["reduce_group_by"] = DEFAULT_GROUP_BY
    if config["reduce_group_by"] not in GROUP_BY_VALUES:
        print >> sys.stderr, 'Attribute "reduce_group_by" must be one of', GROUP_BY_VALUES
        sys.exit(os.EX_CONFIG)

    # =======================================================================
    # == Build job table                                                   ==
    # =======================================================================
    jobs = _build_job_table(config)

    # =======================================================================
    # == MAP                                                               ==
    # =======================================================================
    if options.map:
        ## Load globals
        try:
            user_func.load_globals(config)
        except:
            print >> sys.stderr, "Can not load data"
            sys.exit(os.EX_DATAERR)
        if options.verbose:
            print "** MAP WORKERS TO JOBS **"
        # Use this to load/slice data only once
        resamples_file_cur = resample_nb_cur = None
        data_cur = None
        workers = list()
        for i, job in jobs.iterrows():
            # see if we can create a worker
            while len(workers) == options.ncore:
                # We use a copy of workers to iterate because
                # we remove some element in it
                # See: https://docs.python.org/2/tutorial/datastructures.html#looping-techniques
                for p in workers[:]:
                    #print "Is alive", str(p), p.is_alive()
                    if not p.is_alive():
                        p.join()
                        workers.remove(p)
                        if options.verbose:
                            print "Joined:", str(p)
                time.sleep(1)
            #job = jobs.loc[i]
            try:
                os.makedirs(job[_OUTPUT])
            except:
                if not options.force:
                    continue
            if (not resample_nb_cur and job[_RESAMPLE_INDEX]) or \
               (resample_nb_cur != job[_RESAMPLE_INDEX]):  # Load
                resample_nb_cur = job[_RESAMPLE_INDEX]
                user_func.resample(config, resample_nb_cur)
            key = job[_PARAMS]
            output_collector = job[_OUTPUT_COLLECTOR]
            p = Process(target=user_func.mapper, args=(key, output_collector))
            if options.verbose:
                print "Start :", str(p), str(output_collector)
            p.start()
            workers.append(p)

        for p in workers[:]:  # Join remaining worker
            # Similarly we create a copy of workers to iterate on it
            # while removing elements
            p.join()
            workers.remove(p)
            if options.verbose:
                print "Joined:", str(p)

    if options.clean:
        for i, job in jobs.iterrows():
            output_collector = job[_OUTPUT_COLLECTOR]
            output_collector.clean()

    # =======================================================================
    # == REDUCE                                                            ==
    # =======================================================================
    if options.reduce:
        # Create dict of list of OutputCollectors.
        # It is important that the inner lists are ordered naturally because
        # the reducer generally iterate through this list and some entries may
        # have special meaning (for instance the first fold might be the full
        # population without resampling):
        #  - If grouping by parameters (the most common case) we sort by
        # resample index (i.e. the first fold is the first loaded entry).
        #  - If grouping by resample index, we sort by parameters (i.e. the
        # first tuple of parameters is the first loaded entry)
        # In both cases, the index gives the correct order (see function
        # _build_job_table).
        # Note that by default groupby sorts the key (hence the order of the
        # groups is not and the final CSV is not naturally sorted). Therefore
        # we pass sort=False.
        grouped = jobs.groupby(config["reduce_group_by"], sort=False)
        # Copy the groups in a dictionnary with the same keys than the GroupBy
        # object and the same order. This is needed to sort the groups.
        groups = OrderedDict(iter(grouped))
        n_groups = len(groups)
        ordered_keys = groups.keys()
        # Sort inner groups by index
        for key, group in groups.items():
                group.sort_index(inplace=True)

        if options.verbose:
            print "== Groups found:"
            for key, group in groups.items():
                print key
                print group
        # Do the reduce
        scores_tab = None  # Dict of all the results
        for k in groups:
            try:
                output_collectors = groups[k][_OUTPUT_COLLECTOR]
                # Results for this key
                scores = user_func.reducer(key=k, values=output_collectors)
                # Create df on first valid reducer (we canno't do it before
                # because we don't have the columns).
                # The keys are the keys of the GroupBy object.
                # As we use a df, previous failed reducers (if any) will be
                # empty. Similarly future failed reducers (if any) will be
                # empty.
                if scores_tab is None:
                    index = pd.Index(ordered_keys,
                                     name=config["reduce_group_by"])
                    scores_tab = pd.DataFrame(index=index,
                                              columns=scores.keys())
                # Append those results to scores
                # scores_tab.loc[k] don't work because as k is a tuple
                # it's interpreted as several index.
                # Therefore we use scores_tab.loc[k,].
                # Integer based access (scores_tab.iloc[i]) would work too.
                scores_tab.loc[k,] = scores.values()
            except Exception, e:
#                pass
                print "Reducer failed in {key}".format(key=k), groups[k]
                print "Exception:", e
#        scores = [user_func.reducer(key=k, values=groups[k]) for k in groups]
#        print p.get_open_files()
        if scores_tab is None:
            print >> sys.stderr, "All reducers failed. Nothing saved."
            sys.exit(os.EX_SOFTWARE)
        # Add a columns of glob expressions
        if config["reduce_group_by"] == _PARAMS:
            globs = [os.path.join(config["map_output"],
                                  '*',
                                  param_sep.join([str(p) for p in params]))
                                  for params in scores_tab.index]
        else:
            globs = [os.path.join(config["map_output"],
                                  str(resample),
                                  '*')
                                  for resample in scores_tab.index]
        globs_df = pd.DataFrame(globs,
                                columns=['glob'],
                                 index=index)
        scores_tab = scores_tab.merge(globs_df,
                                      left_index=True,
                                      right_index=True)
        if "reduce_output" in config:
            print "Save results into: %s" % config["reduce_output"]
            scores_tab.to_csv(config["reduce_output"], index=True)
        else:
            print scores_tab.to_string()
