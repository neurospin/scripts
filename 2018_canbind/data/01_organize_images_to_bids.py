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


SRC = os.path.join(WD, "data", "sourcedata", "canbind_mgz")
DST = os.path.join(WD, "data", "sourcedata", "canbind_bids")

# FREESURFER_HOME="/usr/local/freesurfer"
FREESURFER_HOME="/i2bm/local/freesurfer" # At Neurospin
shell = '/bin/bash'

# regexp = re.compile(r"%s/([^/]+)/(.+)_(.+)_(.+)" % SRC)
regexp = re.compile(r"(.+)_(.+)_(.+)")


filenames = glob.glob(os.path.join(SRC, "*", "*.mgz"))
basenames = [os.path.basename(f) for f in filenames]

assert len(filenames) == len(set(basenames)) == 808, "Not unique filename"

items = list()
for filename in filenames:
    # dirname = '/neurospin/psy/canbind/data/raw/canbind_sourcedata_FS/Week8/UCA_0033_03'
    #regexp = re.compile(r"^%s/([^/]+)/(.+)" % SRC)
    src_basename, _ = os.path.splitext(os.path.basename(filename))

    site, sub, tp = regexp.match(src_basename).groups()

    #input_mgz = "/tmp/T1.mgz"
    ouput_dir = os.path.join(DST, "sub-%s-%s" % (site, sub), "ses-%s" % tp, "anat")
    os.makedirs(ouput_dir, exist_ok=True)

    dst_basename = "sub-%s-%s_ses-%s_T1w.nii" % (site, sub, tp)

    ouput_filename = os.path.join(ouput_dir, dst_basename)

    print(filename, ouput_filename)
    items.append(["%s-%s" % (site, sub), site, tp])
    command = ["mri_convert.bin", filename, ouput_filename]
    exec_fs_cmd(command, FREESURFER_HOME="/i2bm/local/freesurfer")



# participants =
df = pd.DataFrame(items, columns=["participant_id", "site", "time_point"])

assert len(glob.glob(os.path.join(DST, "sub-*/ses-*/anat/sub-*_T1w.nii"))) == len(filenames) == df.shape[0] == 808

df["One"] = 1
# df["One"].astype(int)
#df.pivot(index=['participant_id', 'site'], columns="time_point")
#df.pivot(index=['participant_id', 'site'], columns="time_point", values="time_point")
participants = df.pivot(index='participant_id', columns="time_point", values="One").reset_index()

sesssions = ['01', '01-alt03', '01-alt16', '01SE02', '02', '03',  '03SE02']
participants.reset_index()
sesssions = ['ses-'+s for s in sesssions]

participants.columns = ['participant_id'] + sesssions

assert participants[sesssions].sum().sum() == len(items)

print(participants.sum())
#ses-01                                                          258
#ses-01-alt03                                                      1
#ses-01-alt16                                                      1
#ses-01SE02                                                        1
#ses-02                                                          284
#ses-03                                                          262
#ses-03SE02                                                        1
df = df[["participant_id", "site"]].drop_duplicates()
assert df.shape[0] == participants.shape[0] == 310

participants = pd.merge(participants, df, on='participant_id', how='left')
participants["participant_id"] = ["sub-%s" % s for s in participants["participant_id"]]


participants.to_csv(os.path.join(WD, "data", "participants.tsv"), sep='\t', index=False)
ptcp = pd.read_csv(os.path.join(WD, "data", "participants.tsv"), sep="\t")

assert participants.shape[0] == 310
assert participants[sesssions].sum().sum() == 808


