#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:53:01 2018

@author: ed203246


mri_convert all mgz image to niftii with BIDS organization
"""

import os
import os.path
import glob
import re
# https://www.thegeekstuff.com/2014/07/python-regex-examples
# https://www.thegeekstuff.com/2014/07/advanced-python-regex/

import subprocess
import pandas as pd

def exec_fs_cmd(command, FREESURFER_HOME="/i2bm/local/freesurfer"):
    cmd = [shell, '-c', 'export FREESURFER_HOME="{0}"; {0}/SetUpFreeSurfer.sh; exec {0}/bin/{1} '.format(FREESURFER_HOME, command[0]) + \
                        ' '.join("'%s'" % i.replace("'", "\\'") for i in command[1:])]
    out = subprocess.run(cmd, stdout=subprocess.PIPE)
    return out

WD = "/neurospin/psy/canbind"


SRC = os.path.join(WD, "data", "raw", "canbind_sourcedata_FS")
DST = os.path.join(WD, "data", "raw", "canbind")

# FREESURFER_HOME="/usr/local/freesurfer"
FREESURFER_HOME="/i2bm/local/freesurfer" # At Neurospin
shell = '/bin/bash'

regexp = re.compile(r"%s/([^/]+)/(.+)_(.+)_(.+)" % SRC)

dirnames = glob.glob(os.path.join(SRC, "*", "*"))

items = list()
for dirname in dirnames:
    # dirname = '/neurospin/psy/canbind/data/raw/canbind_sourcedata_FS/Week8/UCA_0033_03'
    #regexp = re.compile(r"^%s/([^/]+)/(.+)" % SRC)
    ses, site, sub, tp = regexp.match(dirname).groups()
    input_mgz = os.path.join(dirname, "mri", "T1.mgz")

    #input_mgz = "/tmp/T1.mgz"
    ouput_dir = os.path.join(DST, "sub-%s_%s" % (site, sub), "ses-%s" % ses, "anat")
    os.makedirs(ouput_dir, exist_ok=True)
    ouput_filename = os.path.join(ouput_dir, os.path.basename(dirname) + ".nii")

    print(dirname, ouput_dir)
    items.append(["%s_%s" % (site, sub), site, ses])
    command = ["mri_convert.bin", input_mgz, ouput_filename]
    exec_fs_cmd(command, FREESURFER_HOME="/i2bm/local/freesurfer")

# participants =
df = pd.DataFrame(items, columns=["participant_id", "site", "time_point"])

df["One"] = 1
# df["One"].astype(int)
#df.pivot(index=['participant_id', 'site'], columns="time_point")
#df.pivot(index=['participant_id', 'site'], columns="time_point", values="time_point")
participants = df.pivot(index='participant_id', columns="time_point", values="One").reset_index()

assert participants[['Week0', 'Week2', 'Week8']].sum().sum() == len(items)

print(participants.sum())
# Week0                                                           278
# Week2                                                           262
# Week8                                                           251

participants.to_csv(os.path.join(WD, "data", "participants.tsv"), sep='\t', index=False)
ptcp = pd.read_csv(os.path.join(WD, "data", "participants.tsv"), sep="\t")
assert ptcp[['Week0', 'Week2', 'Week8']].sum().sum() == 791

