# -*- coding: utf-8 -*-
"""
Created on Thu May 29 13:48:54 2014

@author: edouard.duchesnay@cea.fr

Cluster utils
"""

import os
from collections import OrderedDict

##############################################################################
##
# Note that we can't indent this variable as required by PEP 8 because this
# would impact the formating of the PBS files.
job_header = """#!/bin/bash
#PBS -S /bin/bash
"""

opt_string_format = "{option}={value}"

def gabriel_make_sync_data_files(wd, wd_cluster=None, user=None):
    """Create sync_pull.sh and sync_push.sh files in the wd.

    Input
    -----
    wd: string, local working directory

    wd_cluster: string, remote working directory. If None,
    /neurospin/tmp/<user>/basename(wd)

    Ouput
    -----
    String: the path to the files, can be use to do the synchro.

    Example:
    push, pull, wd_cluster = gabriel_make_sync_data_files("/tmp/toto")
    os.system(push)
    """
    import getpass
    if user is None:
        user = getpass.getuser()
    if wd_cluster is None:
        wd_cluster = os.path.join("/neurospin", "tmp", user,
                                  os.path.basename(wd))
    print "# Make sure parent dir of wd_cluster exists"
    cmd = 'ssh %s@gabriel.intra.cea.fr "mkdir %s"' % (user, os.path.dirname(wd_cluster))
    print cmd
    os.system(cmd)
    # preserve:
    # recursive, link, time, group, owner, Devices (scpecial), update, verbose, compress
    push_str = 'rsync -rltgoDuvz %s %s@gabriel.intra.cea.fr:%s/' % (
         wd, user, os.path.dirname(wd_cluster))
    sync_push_filename = os.path.join(wd, "sync_push.sh")
    with open(sync_push_filename, 'wb') as f:
        f.write(push_str)
    os.chmod(sync_push_filename, 0777)
    pull_str = 'rsync -rltgoDuvz %s@gabriel.intra.cea.fr:%s %s/' % (
        user, wd_cluster, os.path.dirname(wd))
    sync_pull_filename = os.path.join(wd, "sync_pull.sh")
    with open(sync_pull_filename, 'wb') as f:
        f.write(pull_str)
    os.chmod(sync_pull_filename, 0777)
    return sync_push_filename, sync_pull_filename, wd_cluster


def gabriel_make_qsub_job_files(output_dirname, cmd, suffix="",
                                nodes=1, mem=None, walltime=None):
    """Build standard PBS files:
     - one for Cati_LowPrio with 12 process per node
     - one for Cati_Long with 12 process per node
     - one for Global_long with 8 process per node

    This is mostly a convenience function for write_job_file.
    The number of nodes, the total job memory and the walltime can be modified.

    Input
    -----

    output_dirname: string, where the files should be created
    (use basename to create the job_name)

    cmd: string, the command to run

    Example
    -------

    gabriel_make_qsub_job_files("/tmp/toto", "mapreduce -map --config /tmp/toto/config.json")
    """
    project_name = os.path.basename(output_dirname)
    job_name = project_name
    options = OrderedDict()
    # Fill options for nodes and ppn
    options['host'] = OrderedDict()
    options['host']['nodes'] = nodes
    if mem is not None:
        options['mem'] = mem
    if walltime is None:
        options['walltime'] = "48:00:00"
    else:
        options['walltime'] = walltime

    queue = "Cati_LowPrio"
    options['host']['ppn'] = 12
    job_filename = os.path.join(output_dirname,
                                'job_%s%s.pbs' % (queue, suffix))
    write_job_file(job_filename=job_filename,
                   job_name=job_name,
                   cmd=cmd,
                   queue=queue,
                   job_options=options)

    queue = "Cati_long"
    options['host']['ppn'] = 12
    job_filename = os.path.join(output_dirname,
                                'job_%s%s.pbs' % (queue, suffix))
    write_job_file(job_filename=job_filename,
                   job_name=job_name,
                   cmd=cmd,
                   queue=queue,
                   job_options=options)

    queue = "Global_long"
    options['host']['ppn'] = 8
    job_filename = os.path.join(output_dirname,
                                'job_%s%s.pbs' % (queue, suffix))
    write_job_file(job_filename=job_filename,
                   job_name=job_name,
                   cmd=cmd,
                   queue=queue,
                   job_options=options)


def write_job_file(job_filename, job_name, cmd, queue, job_options=None):
    """
    Generates a PBS configration file.

    Input
    -----
    job_filename (string): filename

    job_name (string): name of the job

    cmd (string): the command to run

    queue (string): queue on which to run the job

    job_options (dict whose values are either strings or dict): job options.
    If the value is a string, the pair (key, value) will be written on one line
    as key=value.
    If the value is a dict, the key is discarded and the value is written in
    one line as key0=value0:[key1=value1[:...]]. This is useful for options
    like nodes and ppn which must be on the same line.
    You can use OrderedDict to maintain order.
    """
    def opt_string_from_dict(opt_dict):
        opt_strings = []
        for option, value in opt_dict.iteritems():
            opt_strings.append(opt_string_format.format(option=option,
                                                        value=value))
        opt_str = ":".join(opt_strings)
        return opt_str

    with open(job_filename, 'wb') as f:
        f.write(job_header)
        f.write("""#PBS -N %s\n""" % job_name)
        if job_options is not None:
            for option, value in job_options.iteritems():
                if isinstance(value, str):
                    opt_str = opt_string_format.format(option=option,
                                                       value=value)
                if isinstance(value, dict):
                    opt_str = opt_string_from_dict(value)
                f.write("""#PBS -l %s\n""" % opt_str)
        f.write("""#PBS -q %s\n""" % queue)
        f.write("\n")
        f.write("%s\n" % cmd)
    os.chmod(job_filename, 0777)
