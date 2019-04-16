#!/usr/bin/env python3
# -*- coding: utf-8 -*-
epilog = """
SPM segment QC tool. Take a lists of : grey, white matter, csf probability images
and t1 images and plot t1 (--t1 provided) with grey matter density overlay with
contours where GM probability exceed other tissue density.

Also provide csv file of tissue volumes computed as sum(probability map) * voxel size in liter.

Example:
-------

# with csf map
spmsegment_qc.py --gm data/*/c1usub-*.nii.gz --wm data/*/c2usub-*.nii.gz --t1 data/*/usub-*.nii --csf  data/*/c3usub-*.nii.gz --pdf --anat --output withcsf

# without csf map
spmsegment_qc.py --gm data/*/c1usub-*.nii.gz --wm data/*/c2usub-*.nii.gz --t1 data/*/usub-*.nii --pdf --anat --output nocsf

# rsync -rltgoDuvzn ed203246@$NS:/neurospin/psy/hbn/CBIC/derivatives/spmsegment/sub-NDARAM873GAC ./data/

python -W ignore ./src/spmsegment_qc.py --gm data/*/c1usub-*.nii.gz --wm data/*/c2usub-*.nii.gz --t1 data/*/usub-*.nii --csf  data/*/c3usub-*.nii.gz --pdf --anat --output withcsf

"""
import os, sys
import os.path
import argparse
import numpy as np
import nibabel as nib  # import generate a FutureWarning
import matplotlib.pyplot as plt
from nilearn import plotting
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
#

if __name__ == "__main__":
    # Set default values to parameters
    output_dir = os.path.join(os.getcwd(), "slicesdir")
    threshold = 0.01
    alpha_overlay = 0.7
    nslices = 6
    gm_filenames = wm_filenames = csf_filenames = t1_filenames = None
    cut_coords = [-50, -25, 0, 25, 50, 75]

    # parse command line options
    #parser = optparse.OptionParser(description=__doc__)
    parser = argparse.ArgumentParser(epilog=epilog)
    parser.add_argument('--gm', help='list of grey matter proba images (c1)', nargs='+', type=str)
    parser.add_argument('--wm', help='list of white matter proba map images (c2)', nargs='+', type=str)
    parser.add_argument('--csf', help='list of csf proba map images (c3) (optional)', nargs='+', type=str)
    parser.add_argument('-o', '--output', help='Output csv file', type=str)

    options = parser.parse_args()

    ## DEBUG
    """
    options.gm = ["/neurospin/psy/hbn/derivatives/spmsegment/sub-NDARLU606ZDD/c1usub-NDARLU606ZDD_acq-HCP_T1w.nii.gz"]
    options.wm = ["/neurospin/psy/hbn/derivatives/spmsegment/sub-NDARLU606ZDD/c2usub-NDARLU606ZDD_acq-HCP_T1w.nii.gz"]
    options.csf = ["/neurospin/psy/hbn/derivatives/spmsegment/sub-NDARLU606ZDD/c3usub-NDARLU606ZDD_acq-HCP_T1w.nii.gz"]

    options.output = "/neurospin/psy/hbn/analysis/2019_hbn_vbm_predict-ari/data/tissues_volume.csv"

    #
    options.anat = True
    options.pdf = True
    options.nslices = 6
    options.save_seg = False
    """
    ## DEBUG
    if options.gm is None:
        #print("Error: Input is missing.")
        parser.print_help()
        raise SystemExit("Error: grey matter proba images are missing")
    gm_filenames = options.gm

    if options.wm is None:
        #print("Error: Input is missing.")
        parser.print_help()
        raise SystemExit("Error: white matter proba images are missing")
    wm_filenames = options.wm

    if options.csf is None:
        parser.print_help()
        raise SystemExit("Error: csf proba images are missing")

    csf_filenames = options.csf

    assert len(gm_filenames) == len(wm_filenames), "GM and WM list are not of the same length"
    assert len(gm_filenames) == len(csf_filenames), "GM and CSF list are not of the same length"

    #df = pd.DataFrame(dict(gm=gm_filenames, wm=wm_filenames, t1=t1_filenames, csf=csf_filenames))


    def filename_to_participant_id(filename):
        import re
        re_id = re.compile( r"\/(sub-.+)\/")
        return [m for m in re_id.search(filename).groups() if m is not None][0]

    participant_ids = [filename_to_participant_id(filename) for filename in gm_filenames]
    if len(set(participant_ids)) != len(gm_filenames):
        raise SystemExit("participant ID not unique")

    tissues_vol = list()
    # Iterate over images
    for i, participant_id in enumerate(participant_ids):
        # i = 0
        print(participant_id)
        if gm_filenames[i].count(participant_id) < 1:
            raise SystemExit("Missmatch participant_id %s file name %s" % (participant_id, gm_filenames[i]))

        if wm_filenames[i].count(participant_id) < 1:
            raise SystemExit("Missmatch participant_id %s file name %s" % (participant_id, wm_filenames[i]))

        if csf_filenames[i].count(participant_id) < 1:
            raise SystemExit("Missmatch participant_id %s file name %s" % (participant_id, csf_filenames[i]))

        # Load an threshold map, load t1
        gm_img = nib.load(gm_filenames[i])
        voxsize = np.asarray(gm_img.header.get_zooms())
        voxvol = voxsize.prod()  # mm3
        gm_vol = gm_img.get_data().sum() * voxvol / (10 ** 6) # l
        wm_vol = nib.load(wm_filenames[i]).get_data().sum() * voxvol / (10 ** 6) # l
        csf_vol = nib.load(csf_filenames[i]).get_data().sum() * voxvol / (10 ** 6) # l

        tissues_vol.append([participant_id, gm_vol, wm_vol, csf_vol])

    tissues_vol = pd.DataFrame(tissues_vol, columns = ["participant_id", "gm_vol_l", "wm_vol_l", "csf_vol_l"])
    tissues_vol["tiv_l"] = tissues_vol[["gm_vol_l", "wm_vol_l", "csf_vol_l"]].sum(axis=1)
    tissues_vol["GMratio"] = tissues_vol["gm_vol_l"] / tissues_vol["tiv_l"]
    tissues_vol["WMratio"] = tissues_vol["wm_vol_l"] / tissues_vol["tiv_l"]
    tissues_vol["CSFratio"] = tissues_vol["csf_vol_l"] / tissues_vol["tiv_l"]

    tissues_vol.to_csv(options.output, index=False)
