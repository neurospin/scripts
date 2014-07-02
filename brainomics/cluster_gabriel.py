# -*- coding: utf-8 -*-
"""
Created on Thu May 29 13:48:54 2014

@author: edouard.duchesnay@cea.fr

Cluster utils
"""

import os
##############################################################################
##
job_template_pbs =\
"""#!/bin/bash
#PBS -S /bin/bash
#PBS -N %(job_name)s
#PBS -l nodes=1:ppn=%(ppn)s
#PBS -l walltime=48:00:00
#PBS -q %(queue)s

%(script)s
"""
#PBS -d %(job_dir)s

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


def gabriel_make_qsub_job_files(output_dirname, cmd):
    """Build PBS files, one for Cati_LowPrio 12ppn, and one for Global_long
    8ppn

    Input
    -----

    output_dirname: string, where the files should be created
    (use basename to create the job_name)

    cmd: string the command to run

    Example
    -------

    gabriel_make_qsub_job_files("/tmp/toto", "mapreduce -map --config /tmp/toto/config.json")
    """
    project_name = os.path.basename(output_dirname)
    #config_filename = os.path.join(wd_cluster, config_basename)
    #job_dir = os.path.dirname(config_filename)
    #for nb in xrange(options.pbs_njob):
    params = dict()
    params['job_name'] = '%s' % project_name
    params['script'] =  cmd
    params['ppn'] = 12
    params['queue'] = "Cati_LowPrio"
    qsub = job_template_pbs % params
    job_filename = os.path.join(output_dirname, 'job_Cati_LowPrio.pbs')
    with open(job_filename, 'wb') as f:
        f.write(qsub)
    os.chmod(job_filename, 0777)
    params['queue'] = "Cati_long"
    qsub = job_template_pbs % params
    job_filename = os.path.join(output_dirname, 'job_Cati_long.pbs')
    with open(job_filename, 'wb') as f:
        f.write(qsub)
    os.chmod(job_filename, 0777)
    params['ppn'] = 8
    params['queue'] = "Global_long"
    qsub = job_template_pbs % params
    job_filename = os.path.join(output_dirname, 'job_Global_long.pbs')
    with open(job_filename, 'wb') as f:
        f.write(qsub)
    os.chmod(job_filename, 0777)
