import numpy as np
import nibabel
from nilearn.image import resample_img
import pandas as pd
import os, argparse
from tqdm import tqdm
from skimage.transform import rescale

def nilearn_resample_all(participants_path, factor=2.0, **kwargs):
    nii_filenames = pd.read_csv(participants_path, **kwargs).ni_path
    if len(nii_filenames) == 0:
        return []
    ref_img = nibabel.load(nii_filenames[0])
    target_affine = np.copy(ref_img.affine)[:3, :][:, :3]
    target_affine[:3, :3] *= factor
    ref_img_resampled = resample_img(ref_img, target_affine=target_affine)
    data = np.zeros((len(nii_filenames), 1)+ref_img_resampled.shape, dtype=np.float32)
    print('Ref img of shape: {}\t After resizing: {}\tData size after resizing: {}'.format(ref_img.shape, ref_img_resampled.shape,
                                                                                data.shape))
    pbar = tqdm(total=len(nii_filenames), desc='# Img resampled')
    for i, filename in enumerate(nii_filenames):
        img = nibabel.load(filename)
        assert np.all(img.affine == ref_img.affine), "Got referential {} while expecting {}".format(img.affine, ref_img.affine)
        assert np.all(img.shape == ref_img.shape), "Got shape {} while expecting {}".format(
            img.shape, ref_img.shape)
        img_resampled = resample_img(img, target_affine=target_affine)
        data[i, 0] = img_resampled.get_data()
        pbar.update()
    return data


def sk_resample_all(ni_arr, factor=2.0, dtype=np.float32, **kwargs):
    if len(ni_arr) == 0:
        return []
    ni_ref = ni_arr[0]
    ni_ref_resampled = rescale(ni_ref, factor, **kwargs)
    arr_resampled = np.zeros((len(ni_arr),)+ni_ref_resampled.shape, dtype=dtype)
    print('Ref img of shape: {}\t After resizing: {}\tData size after resizing: {}'.format(ni_ref.shape,
                                                                                           ni_ref_resampled.shape,
                                                                                           arr_resampled.shape),
          flush=True)
    pbar = tqdm(total=len(ni_arr), desc='# Img resampled')
    for i, arr in enumerate(ni_arr):
        arr_resampled[i] = rescale(arr, factor, **kwargs)
        pbar.update()
    return arr_resampled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--py_path', type=str, required=True)
    parser.add_argument('--resampling_factor', type=float, required=True)
    parser.add_argument('--output_path', type=str, required=True)


    args = parser.parse_args()

    ni_arr = np.load(args.py_path, mmap_mode='r')
    resampled_data = sk_resample_all(ni_arr, factor=1.0/float(args.resampling_factor), dtype=np.float32)
    new_py_path = os.path.splitext(args.py_path)[0].replace("data64", "data32")
    np.save(new_py_path+'_%.1fmm_skimage.npy'%args.resampling_factor, resampled_data)


    ## RESAMPLING to 1.5mm3
    #!/bin/bash
    # MAIN=~/PycharmProjects/neurospin/scripts/2019_psy_datasets/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/generic_quasi_raw_preproc.py
    # MAIN_RESAMPLING=~/PycharmProjects/neurospin/scripts/2019_psy_datasets/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/quasi_raw_resample.py
    # OUTPUT_PATH=/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/quasi_raw/
    # for DATASET in mpi-leipzig cnp abide1 abide2 ixi icbm localizer candi gsp corr bsnip schizconnect-vip npc biobd hcp nar rbp
    # do
    #  DATA_PATH=$OUTPUT_PATH$DATASET'_t1mri_quasi_raw_data64.npy'
    #  python3 $MAIN_RESAMPLING --py_path $DATA_PATH --resampling_factor 1.5 --output_path $OUTPUT_PATH &> $DATASET'_resampling.txt' &
    # done