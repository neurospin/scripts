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

_RESAMPLE_KEY = 'resample_key'
_PARAMS = 'params'
_OUTPUT = 'output_dir'
_OUTPUT_COLLECTOR = 'output collector'

GROUP_BY_VALUES = [_RESAMPLE_KEY, _PARAMS]
DEFAULT_GROUP_BY = _PARAMS

# Default values for resample and params
_NULL_RESAMPLE = {0: None}
_NULL_PARAMS = {"void": tuple()}

# Global data
DATA = dict()
param_sep = "_"


# Detailed help topics

execution = """
Execution
---------

The script calls functions defined in a separate script. Before calling them,
the script will cd to the folder of the config file.

load_globals(config) is executed once at the beginning to load the data and
define constants.

In map mode:
    for each resampling:
        call resample(config, resample_nb)
        for each param in parameter:
            call mapper(param)
The program can use multiple cores to paralellize mappers.
If the output directory for a given mapper already exists, it will be skipped
(this allows parallelization between several computers with shared filesystem).

In reduce mode:
    output of mappers are grouped (either by resampling or by parameter)
    for each key, list_of_values in group of output:
        call reducer(key, list_of_values)
Note that reduce mode is optionnal.

Output hierarchy is organized as follows:
    <map_output>/<resample_nb>/<params>
If no resampling is provided the output is organized as follows:
    <map_output>/0/<params>
If no parameters are provided the output is organized as follows:
     <map_output>/<resample_nb>/void
If no parameters and no resampling are provided the script stops.

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
    "resample": (list) list of resamplings.
        resample will be called for each value in this list
        Ex: for cross-validation like resampling, use a list of list of list of
            indices like [[[0, 2], [1, 3]], [[1, 3], [0, 2]]].
        Ex: for bootstraping/permutation like resampling, use list of list of
            indices, like [[0, 1, 2, 3], [1, 3, 0, 2]].
    "params":  (list) list of parameters values.
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


def _build_job_table(map_output, resamplings, parameters):
    """Build a pandas dataframe representing the jobs.
    The dataframe has 3 columns whose name is given by global variables:
      - _RESAMPLE_KEY: the index of the resampling
      - _PARAMS: the key passed to map (tuple of parameters)
      - _OUTPUT: the output directory
      - _OUTPUT_COLLECTOR: the OutputCollector
    In order to be able to group by parameters, they must be hashable (it's the
    case for tuples made of strings and floats).
    Note that the index respects the natural ordering of (resample, params) as
    given in the config file.
    """
    # Check that we have resamplings; otherwise fake it
    if resamplings is None:
        resamplings = _NULL_RESAMPLE

    # Check that we have parameters; otherwise fake it
    if parameters is None:
        parameters = _NULL_PARAMS

    # The parameters are given as list of values.
    jobs = [[resample_key,
             parameters[params_key],
             os.path.join(map_output,
                          str(resample_key),
                          str(params_key))]
            for resample_key in resamplings.keys()
            for params_key in parameters.keys()]
    jobs = pd.DataFrame.from_records(jobs,
                                     columns=[_RESAMPLE_KEY,
                                              _PARAMS,
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

    # Check that we have at least resample or params
    if ("resample" not in config) and ("params" not in config):
        print >> sys.stderr, 'Attributes "resample" or "params" are required'
        sys.exit(os.EX_CONFIG)

    # Check that we have map_output
    if "map_output" not in config:
        print >> sys.stderr, 'map_output" is required'
        sys.exit(os.EX_CONFIG)

    # =======================================================================
    # == Build job table                                                   ==
    # =======================================================================
    if "resample" in config:
        do_resampling = True
        resamplings = config["resample"]

        # If resample is not a dict make it as a dict (the key is given by the
        # index).
        if not isinstance(resamplings, dict):
            resamplings = {
                i: resampling for i, resampling in enumerate(resamplings)}
    else:
        do_resampling = None
        resamplings = None

    if "params" in config:
        params = config["params"]
        # If params are given as a file, load them
        params = json.load(open(params)) \
            if isinstance(params, str) else params

        # If params is not a dict make it as a dict (the key is given by the
        # path for each parameter set).
        if not isinstance(params, dict):
            params = {
                param_sep.join([str(p) for p in param]): param
                for param in params}

        # Cast values to tuples
        keys = params.keys()
        for key in keys:
            params[key] = tuple(params[key])
    else:
        params = None

    jobs = _build_job_table(config["map_output"], resamplings, params)

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
            print "** MAP WORKERS TO JOBS **"
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
                            print "Joined:", str(p)
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
        # Check that we have reduce_group_by or use default value
        if "reduce_group_by" not in config:
            config["reduce_group_by"] = DEFAULT_GROUP_BY
        if config["reduce_group_by"] not in GROUP_BY_VALUES:
            print >> sys.stderr, 'Attribute "reduce_group_by" ', \
                "must be one of", GROUP_BY_VALUES, "or absent"
            sys.exit(os.EX_CONFIG)

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
        # object and the same order. This is needed to sort the groups only
        # once.
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
            output_collectors = groups[k][_OUTPUT_COLLECTOR]
                # Results for this key
            scores = user_func.reducer(key=k, values=output_collectors)
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
                index = pd.Index(ordered_keys,
                                 name=config["reduce_group_by"],
                                 tupleize_cols=False)
                scores_tab = pd.DataFrame(index=index,
                                          columns=scores.keys())
            # Append those results to scores
            # scores_tab.loc[k] don't work because as k is a tuple
            # it's interpreted as several index.
            # Therefore we use scores_tab.loc[k,].
            # Integer based access (scores_tab.iloc[i]) would work too.
            scores_tab.loc[k, ] = scores.values()
        if scores_tab is None:
            print >> sys.stderr, "All reducers failed. Nothing saved."
            sys.exit(os.EX_SOFTWARE)
        if "reduce_output" in config:
            print "Save results into: %s" % config["reduce_output"]
            scores_tab.to_csv(config["reduce_output"], index=True)
        else:
            print scores_tab.to_string()
