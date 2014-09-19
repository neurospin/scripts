# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 18:43:11 2014

@author: md238665

Create a simple exemple to trace the process

"""


import os
#import sys
import tempfile
import time
import random
import json


def load_globals(config):
    pass


def resample(config, resample_nb):
    pass


def info(title):
    print title
    print 'module name:', __name__
    print 'parent process:', os.getppid()
    print 'process id:', os.getpid()


def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables
    t, = key
    #stdout = os.path.join(output_collector.output_dir,
    #                      str(os.getpid()) + ".out")
    #sys.stdout = open(stdout, "w")
    #info('function mapper')
    pid = os.getpid()
    sleep_time = float(t + random.random())
    time.sleep(sleep_time)
    output_collector.collect(key, dict(pid=pid, sleep_time=sleep_time))


def reducer(key, values):
    # values are OutputCollectors containing a path to the results.
    # load return dict correspondning to mapper ouput. they need to be loaded.
    return dict(param=key)


if __name__ == "__main__":
    WD = tempfile.mkdtemp()

    # mapreduce will set its WD to the directory that contains the config file
    # use relative path
    user_func_filename = os.path.abspath(__file__)
    config = dict(data={},
                  params=[[1.0], [5.0], [10.0], [50.0]],
                  resample=[[0]],
                  map_output="results",
                  user_func=user_func_filename,
                  reduce_group_by="params_str",
                  reduce_output="results.csv")
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))
    exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                      "..", "mapreduce.py"))
    ###########################################################################
    ## Apply map
    map_cmd = "%s --map %s/config.json --ncore 2" % (exec_path, WD)
    reduce_cmd = "%s -v --reduce %s/config.json" % (exec_path, WD)
    os.system(map_cmd)
    os.system(reduce_cmd)
