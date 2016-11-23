#!/usr/bin/env python
"""
Tool to resample data and explore a parameter grid based on the MapReduce
paradigm.
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


#############
# Constants #
#############

_RESAMPLE = 'resample'
_RESAMPLE_KEY = 'resample_key'
_PARAMS = 'params'
_OUTPUT = 'output_dir'
_OUTPUT_COLLECTOR = 'output collector'

GROUP_BY_VALUES = [_RESAMPLE_KEY, _PARAMS]
DEFAULT_GROUP_BY = _PARAMS

# Default values for resample and params
_NULL_RESAMPLING = {0: None}
_NULL_PARAMS = {"void": tuple()}

# Global data
DATA = dict()
param_sep = "_"


# Detailed help topics

execution = """
Execution
---------

The script calls functions defined in a separate script. Before calling them,
the script will cd to the folder of the config file. Four functions can be
defined: load_globals, resample, mapper and reducer. Only mapper is mandatory.

There are 2 modes: map and reduce (controlled by CLI arguments).

Execution is as follows:

First, load_globals(config) is executed once at the beginning to load the data
and define constants.

In map mode:
    for each resampling:
        call resample(config, resample_key)
        for each param in parameter:
            call mapper(param)
The program can use multiple cores to parallelize mappers.
If the output directory for a given mapper already exists, it will be skipped
(this allows parallelization between several computers with shared filesystem).

In reduce mode:
    output of mappers are grouped (either by resampling or by parameter)
    for each key, list_of_values in group of output:
        call reducer(key, dict_of_values)
dict_of_values is a dict with keys given by the grouping parameter.
Note that reduce mode is optionnal.

Output hierarchy is organized as follows:
    <map_output>/<resample_key>/<params_key>
If no resampling is provided the output is organized as follows:
    <map_output>/0/<params_key>
If no parameters are provided the output is organized as follows:
     <map_output>/<resample_key>/void
If no parameters and no resamplings are provided the script stops.

The resamplings are given in a iterable or dictionary (if it is an iterable, it
will be converted to a dict with the index as key). The key is used for the
directory.
Parameters are given in a list or dict. The key is used for the directory. If
absent (i.e. when the parameters are given in a list) it is generated from the
values.
In order to be able to group by parameters, they must be hashable (it's the
case for tuples made of strings and floats).

"""

config_file = """
Config file
-----------

The config file is a JSON-encoded dictionary.
There are 3 required entries:
    "map_output": (string) root directory of mappers output.
    "user_func": (string) path to a python file that contains the user defined
        functions.
    "reduce_group_by": (string; values """ + str(GROUP_BY_VALUES) + """,
        default '""" + DEFAULT_GROUP_BY + """'). Required only in reduce mode.

Moreover at least one of the following entries is needed:
    "resample": (list or dict) list of resamplings.
        resample will be called for each value in this list/dict.
        Ex: for cross-validation like resampling, use a list of list of list of
            indices like [[[0, 2], [1, 3]], [[1, 3], [0, 2]]].
        Ex: for bootstraping/permutation like resampling, use list of list of
            indices, like [[0, 1, 2, 3], [1, 3, 0, 2]].
    "params":  (list or dict) list of parameters values.
        mapper will be called for each value in this list (after resampling).
        The value is often a list of values that will be interpreted in mapper.
        Ex: [[1.0, 0.1], [1.0, 0.9], [0.1, 0.1], [0.1, 0.9]].

Other optional values:
    "reduce_output": (string) path where to store the reducer output
        (CSV format). If not specified, output to stdout.

Other fields can be included to be used by functions. For example there is
often a field called "data" which contains the relative path to the data that
is used in load_globals (see example).
"""

example = """
Example
-------

## Create dataset
n, p = 10, 5
X = np.random.rand(n, p)
beta = np.random.rand(p, 1)
y = np.dot(X, beta)
np.save("X.npy", X)
np.save("y.npy", y)

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
              reduce_group_by="params")
json.dump(config, open("config.json", "w"))

"""

epilog = "\n".join([execution, config_file, example])

#############
# Functions #
#############


def load_data(key_filename):
    return {key: np.load(key_filename[key]) for key in key_filename}


def dir_from_param_list(param_list):
    return param_sep.join([str(p) for p in param_list])


def _build_job_table(map_output, resamplings, parameters, compress,
                     force_pickle):
    """Build a pandas dataframe representing the jobs.
    The dataframe has 4 columns whose names are given by global variables:
      - _RESAMPLE_KEY: the index of the resampling
      - _PARAMS: the key passed to map (tuple of parameters)
      - _OUTPUT: the output directory
      - _OUTPUT_COLLECTOR: the OutputCollector
    """
    jobs = [[resample_key,
             parameters[params_key],
             os.path.join(map_output,
                          str(resample_key),
                          str(params_key))]
            for resample_key in list(resamplings.keys())
            for params_key in list(parameters.keys())]
    jobs = pd.DataFrame.from_records(jobs,
                                     columns=[_RESAMPLE_KEY,
                                              _PARAMS,
                                              _OUTPUT])
    # Add OutputCollectors (we need all the rows so we do that latter)
    jobs[_OUTPUT_COLLECTOR] = jobs[_OUTPUT].map(
        lambda x: OutputCollector(x, compress, force_pickle))
    return jobs


def _makedirs_safe(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def import_module_from_filename(filename):
    sys.path.append(os.path.dirname(filename))
    name, _ = os.path.splitext(os.path.basename(filename))
    user_module = __import__(os.path.basename(name))
    return user_module

_import_user_func = import_module_from_filename


# TODO: add different error code for nonexistent dir and empty dir cases
class MapperError(Exception):
    """Dummy class to represent exceptions caused by failed mapper jobs."""
    pass


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
    def __init__(self, output_dir, compress=True, force_pickle=True):
        self.output_dir = output_dir
        self.compress = compress
        self.force_pickle = force_pickle

    def clean(self):
        if os.path.exists(self.output_dir) \
                and (len(os.listdir(self.output_dir)) == 0):
            print("clean", self.output_dir)
            os.rmdir(self.output_dir)

    def collect(self, key, value):
        _makedirs_safe(self.output_dir)
        for k in value:
            if isinstance(value[k], np.ndarray):
                if self.compress:
                    np.savez_compressed(os.path.join(self.output_dir, k),
                                        value[k])
                else:
                    np.save(os.path.join(self.output_dir, k), value[k])
            elif isinstance(value[k], nibabel.Nifti1Image):
                filename = os.path.join(self.output_dir, k + ".nii")
                if self.compress:
                    filename = filename + ".gz"
                value[k].to_filename(filename)
            else:
                if self.force_pickle:
                    of = open(os.path.join(self.output_dir, k + ".pkl"), "w")
                    pickle.dump(value[k], of)
                    of.close()
                else:
                    try:
                        of = open(os.path.join(self.output_dir, k + ".json"),
                                  "w")
                        json.dump(value[k], of)
                        of.close()
                    except:
                        of = open(os.path.join(self.output_dir, k + ".pkl"),
                                  "w")
                        pickle.dump(value[k], of)
                        of.close()

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.output_dir)

    def load(self, pattern="*"):
        #print os.path.join(self.output_dir, pattern)
        res = dict()
        if not os.path.exists(self.output_dir):
            msg = "Output dir not found ({dir})".format(dir=self.output_dir)
            raise MapperError(msg)
        files = glob.glob(os.path.join(self.output_dir, pattern))
        if len(files) == 0:
            msg = "Output dir empty ({dir})".format(dir=self.output_dir)
            raise MapperError(msg)
        for filename in files:
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
        description=__doc__, epilog=epilog)

    parser.add_argument('-m', '--map', action='store_true', default=False,
                        help="Run mapper: iterate over resamples and "
                        "parameters, and call mapper (defined in user_func)")

    parser.add_argument('-c', '--clean', action='store_true', default=False,
                        help="Clean execution: remove empty directories "
                        "in map_output. Use it if you suspect that some "
                        "mapper jobs did not end not properly.")

    parser.add_argument('-r', '--reduce', action='store_true', default=False,
                        help="Run reducer: iterate over map_output and call "
                        "reduce (defined in user_func)")

    parser.add_argument('--raw', action='store_true', default=False,
                        help="Don't use compression for nibabel images and "
                        "numpy array (default: use compression)")

    parser.add_argument('-j', '--json', action='store_true', default=False,
                        help="Try to serialize data with JSON first "
                        "(default: try pickle first)")

    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help="Force call to mapper even if output is present")

    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    # Config file
    parser.add_argument('config', help='Configuration json file that '
                        'contains a dictionary of configuration options.')

    default_nproc = cpu_count()
    parser.add_argument(
        '--ncore',
        help='Nb of cpu cores to use (default %i)' % default_nproc, type=int)
    options = parser.parse_args()

    if not options.config:
        print('Required config file', file=sys.stderr)
        sys.exit(os.EX_USAGE)
    config_filename = options.config

    # Check that at least one mode is given
    if not (options.map or options.reduce or options.clean):
        print('Require a mode (map, reduce or clean)', file=sys.stderr)
        sys.exit(os.EX_USAGE)
    # Check that only one mode is given
    if (options.map and options.reduce) or \
       (options.map and options.clean) or \
       (options.reduce and options.clean):
        print('Require only one mode (map, reduce or clean)', file=sys.stderr)
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
        print('Attribute "user_func" is required', file=sys.stderr)
        sys.exit(os.EX_CONFIG)
    user_func = _import_user_func(config["user_func"])

    # Check that we have at least resample or params
    if ("resample" not in config) and ("params" not in config):
        print('Attributes "resample" or "params" are required', file=sys.stderr)
        sys.exit(os.EX_CONFIG)

    # Check that we have map_output
    if "map_output" not in config:
        print('map_output" is required', file=sys.stderr)
        sys.exit(os.EX_CONFIG)

    # =======================================================================
    # == Build job table                                                   ==
    # =======================================================================
    if "resample" in config:
        do_resampling = True
        resamplings = config["resample"]

        # If resample is not a dict make it as a dict (the key is given by the
        # index). We keep the order here.
        if not isinstance(resamplings, dict):
            resamplings = OrderedDict([
                (i, resampling) for i, resampling in enumerate(resamplings)])
    else:
        do_resampling = None
        resamplings = _NULL_RESAMPLING

    if "params" in config:
        params = config["params"]
        # If params are given as a file, load them
        params = json.load(open(params)) \
            if isinstance(params, str) else params

        # If params is not a dict make it as a dict (the key is given by the
        # path for each parameter set). We keep the order here.
        if not isinstance(params, dict):
            params = OrderedDict([(dir_from_param_list(param), param)
                                  for param in params])

        # Cast values to tuples
        keys = list(params.keys())
        for key in keys:
            params[key] = tuple(params[key])
    else:
        params = _NULL_PARAMS

    jobs = _build_job_table(config["map_output"], resamplings, params,
                            compress=not options.raw, 
                            force_pickle=not options.json)

    # =======================================================================
    # == Load globals                                                      ==
    # =======================================================================
    if (options.map) or (options.reduce):
            user_func.load_globals(config)
#        try:
#            user_func.load_globals(config)
#        except Exception as e:
#                print >> sys.stderr, "Cannot load data"
#                print >> sys.stderr, e.__class__.__name__, "exception:", e
#                sys.exit(os.EX_DATAERR)

    # =======================================================================
    # == MAP                                                               ==
    # =======================================================================
    if options.map:
        if options.verbose:
            print("** MAP WORKERS TO JOBS **")
        # Use this to load/slice data only once
        resamples_file_cur = resample_key_cur = None
        data_cur = None
        workers = list()
        for i, job in jobs.iterrows():
            # see if we can create a worker
            while len(workers) == options.ncore:
                # We use a copy of workers to iterate because
                # we remove some element in it
                # See: https://docs.python.org/2/tutorial/datastructures.html
                # (section "Looping Techniques")
                for p in workers[:]:
                    #print "Is alive", str(p), p.is_alive()
                    if not p.is_alive():
                        p.join()
                        workers.remove(p)
                        if options.verbose:
                            print("Joined:", str(p))
                time.sleep(1)
            #job = jobs.loc[i]
            try:
                os.makedirs(job[_OUTPUT])
            except:
                if not options.force:
                    continue
            if do_resampling:
                if (not resample_key_cur and job[_RESAMPLE_KEY]) or \
                   (resample_key_cur != job[_RESAMPLE_KEY]):  # Load
                    resample_key_cur = job[_RESAMPLE_KEY]
                    user_func.resample(config, resample_key_cur)
            key = job[_PARAMS]
            output_collector = job[_OUTPUT_COLLECTOR]
            p = Process(target=user_func.mapper, args=(key, output_collector))
            if options.verbose:
                print("Start :", str(p), str(output_collector))
            p.start()
            workers.append(p)

        for p in workers[:]:  # Join remaining worker
            # Similarly we create a copy of workers to iterate on it
            # while removing elements
            p.join()
            workers.remove(p)
            if options.verbose:
                print("Joined:", str(p))

    if options.clean:
        for i, job in jobs.iterrows():
            output_collector = job[_OUTPUT_COLLECTOR]
            output_collector.clean()

    # =======================================================================
    # == REDUCE                                                            ==
    # =======================================================================
    if options.reduce:
        # Check that we have reduce_group_by or use default value
        if "reduce_group_by" not in config:
            config["reduce_group_by"] = DEFAULT_GROUP_BY
        if config["reduce_group_by"] not in GROUP_BY_VALUES:
            print('Attribute "reduce_group_by" ', \
                "must be one of", GROUP_BY_VALUES, "or absent", file=sys.stderr)
            sys.exit(os.EX_CONFIG)
        outer_key = config["reduce_group_by"]
        outer_key_index = GROUP_BY_VALUES.index(outer_key)
        # If outer_key is params, this will yield resampling_key and
        # vice-versa.
        inner_key = GROUP_BY_VALUES[outer_key_index - 1]

        # Group outputs by outer_key
        grouped = jobs.groupby(outer_key)
        index = list(grouped.groups.keys())  # Index of results DataFrame
        if options.verbose:
            print("== Groups found:")
            for key, group in grouped:
                print(key)
                print(group)

        # Do the reduce (we can no longer guarantee the order of output)
        scores_tab = None  # DataFrame of all the results
        for key, group in grouped:
            # Create dict of OutputCollectors
            group = group.set_index(inner_key)
            output_collectors = group[_OUTPUT_COLLECTOR].to_dict()
            # Call reducer for this key
            scores = user_func.reducer(key=key, values=output_collectors)
            """ Do not catch exceptions
            try:
                output_collectors = groups[k][_OUTPUT_COLLECTOR]
                # Results for this key
                scores = user_func.reducer(key=k, values=output_collectors)
            except MapperError as e:
                print "Reducer failed in {key} because it can't access " \
                      "data.".format(key=k)
                print "Exception:", e
                print "This is probably because the mapper failed."
                continue
            except Exception as e:
                print >> sys.stderr, "Reducer failed in {key}".format(key=k)
                print >> sys.stderr, e.__class__.__name__, "exception:", e
                sys.exit(os.EX_SOFTWARE)
            """
            # Create df on first valid reducer (we cannot do it before
            # because we don't have the columns).
            # The keys are the keys of the GroupBy object.
            # As we use a df, previous failed reducers (if any) will be
            # empty. Similarly future failed reducers (if any) will be
            # empty.
            if scores_tab is None:
                # tupleize_cols was introduced in pandas 0.14 to automatically
                # create MultiIndex. For now we force it to False in 0.14 (it
                # is ignored in previous versions).
                index = pd.Index(index,
                                 name=outer_key,
                                 tupleize_cols=False)
                scores_tab = pd.DataFrame(index=index,
                                          columns=list(scores.keys()))
            # Append those results to scores
            # scores_tab.loc[k] don't work because as k is a tuple
            # it's interpreted as several index.
            # Therefore we use scores_tab.loc[k,].
            # Integer based access (scores_tab.iloc[i]) would work too.
            scores_tab.loc[key, ] = list(scores.values())
        if scores_tab is None:
            print("All reducers failed. Nothing saved.", file=sys.stderr)
            sys.exit(os.EX_SOFTWARE)
        if "reduce_output" in config:
            print("Save results into: %s" % config["reduce_output"])
            scores_tab.to_csv(config["reduce_output"], index=True)
        else:
            print(scores_tab.to_string())
