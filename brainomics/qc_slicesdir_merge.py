#!/usr/bin/env python3
# -*- coding: utf-8 -*-

epilog = """
Merge output of fsl slicesdir into one html file

Example
-------

cd /neurospin/psy/canbind/data/derivatives/spmsegment/QC
~/git/scripts/brainomics/qc_slicesdir_merge.py --slicesdirs "slicesdir_raw slicesdir_rc1" --subjects ../LIST_SUBJECTS/list_vbm_T1.txt --output slices_raw-gm.html
"""

import glob
import os
import re
import numpy as np
import pandas as pd
import argparse

"""
list_scans_filenames = "/neurospin/psy/canbind/data/derivatives/spmsegment/LIST_SUBJECTS/list_vbm_T1.txt"

input_slicesdirs = [
'/neurospin/psy/canbind/data/derivatives/spmsegment/QC/slicesdir_raw',
'/neurospin/psy/canbind/data/derivatives/spmsegment/QC/slicesdir_rc1']

output = '/tmp/index.html'
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--slicesdirs', help='Input directories from fsl slicesdir', type=str)
    parser.add_argument('--subjects', help='file containing the fist of subjects where basename respect BIDS: sub-<ID>_ses-<time point>_<modality>.nii', type=str)
    parser.add_argument('-o', '--output', help='Output html file', type=str)
    options = parser.parse_args()

    if options.slicesdirs is None:
            #print("Error: Input is missing.")
            parser.print_help()
            raise SystemExit("Error: Input fsl input_slicesdirs are missing.")
    slicesdirs = options.slicesdirs.split()
    #print(slicesdirs)

    if options.subjects is None:
            #print("Error: Input is missing.")
            parser.print_help()
            raise SystemExit("Error: Input fsl slicesdir are missing.")
    list_scans_filenames = options.subjects

    if options.output is None:
            #print("Error: Input is missing.")
            parser.print_help()
            raise SystemExit("Error: output is missing.")
    output = options.output

    #print(list_scans_filenames, slicesdirs)

    with open(list_scans_filenames, 'r') as fd:
        scans = [os.path.splitext(os.path.basename(f.strip()))[0] for f in fd.readlines()]

    qc_tab = pd.DataFrame([s.split('_') for s in scans], columns = ['participant_id', 'session', 'modality'])

    scans = {s:[] for s in scans}

    for dir_ in slicesdirs:
        #dir_ = dirs[0]
        filenames = glob.glob(os.path.join(dir_, "*sub*.png"))
        assert len(filenames) == len(scans), "Found %i files in %s != #subjects %i in %s" % (len(filenames), dir_, len(scans), list_scans_filenames)
        for filename in filenames:
            #filename = filenames[0]
            found = [scan for scan in scans if filename.count(scan)]
            assert len(found) == 1
            scans[found[0]].append(filename)

    assert np.all(np.array([len(v) for v in scans.values()]) == len(slicesdirs))

    ofd = open(output, "w")

    for k in scans:
        ofd.write('<HTML><TITLE>slicedir</TITLE><BODY BGCOLOR="#aaaaff">' + '\n')

        for i in range(len(scans[k])):
            line = '<a href="%s"><img src="%s" WIDTH=1000 > %s</a><br>' % (scans[k][i], scans[k][i], k)
            ofd.write(line + '\n')
        ofd.write('<br>\n')

    ofd.write('</BODY></HTML>' + '\n')
    ofd.close()

    qc_tab.to_csv(os.path.splitext(output)[0] + "_qc.csv", index=False)
