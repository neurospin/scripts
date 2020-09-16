import numpy as np
import nibabel
from nilearn.image import resample_img
import pandas as pd
import os
from tqdm import tqdm

def resample_all(participants_path, factor=2.0, **kwargs):
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


if __name__ == "__main__":

    participants_path = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/' \
                 'data/quasi_raw/all_t1mri_quasi_raw_participants.tsv'

    resampled_data = resample_all(participants_path, factor=1.5, sep='\t')
    np.save(os.path.join(os.path.dirname(participants_path), 'all_t1mri_quasi_raw_data32_1.5mm.npy'), resampled_data)
