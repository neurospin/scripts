import os
import glob
import numpy as np
from skimage import morphology
import nibabel as nib
from skimage.morphology import disk, square
import argparse

def is_dir(dirarg):
    """ Type for argparse - checks that output dir exists.
    """
    if not os.path.isdir(dirarg):
        raise argparse.ArgumentError(
            "The dir '{0}' does not exist!".format(dirarg))
    return dirarg

parser = argparse.ArgumentParser()
parser.add_argument('-i1', '--image1', metavar='FILE', required=False,
                    help='Image T1 to correct for bias field')
parser.add_argument('-i2', '--image2', metavar='FILE', required=False,
                    help='Image T2 or FLAIR to correct for bias field')
parser.add_argument('-m', '--mask', metavar='FILE', required=True,
                    help='mask of slices obtained from MNI')
parser.add_argument('-d', '--outdir', metavar='PATH', required=True,
                    type=is_dir,
                    help='Output directory to create the file in.')


def creation_mask(filename_mask, filename_back_T1, filename_back_T2, type):
    img = nib.load(filename_mask)
    back_T2 = nib.load(filename_back_T2)
    back_T1 = nib.load(filename_back_T1)
    id_patient = filename_mask[45:-62]
    base_directory = filename_mask[0:-74]

    if type == "square":
        selem = square(3)
    else:
        selem = disk(1)

    img_data = img.get_data()
    img_data_1 = np.nan_to_num(img_data * ((img_data == 1) + (img_data == 500)))
    img_data_500 = np.nan_to_num(img_data * (img_data == 500))
    img_data_1000 = np.nan_to_num(img_data * ((img_data == 1000) + (img_data == 500)))

    img_data_clsd_z_1 = img_data_1
    img_data_clsd_z_1000 = img_data_1000

    for j in range(img_data_clsd_z_1000.shape[2]):
        img_data_clsd_z_1000[:, :, j] = morphology.binary_closing(img_data_1000[:, :, j], selem)
    for k in range(img_data_clsd_z_1.shape[2]):
        img_data_clsd_z_1[:, :, k] = morphology.binary_closing(img_data_1[:, :, k], selem)

    img_data_clsd_z_500 = img_data_clsd_z_1 * img_data_clsd_z_1000
    img_data_clsd_z_1_superposable = img_data_clsd_z_1 * ((img_data_clsd_z_500 == 0) * (img_data_clsd_z_1 == 1))
    img_data_clsd_z_1000_superposable = img_data_clsd_z_1000 * ((img_data_clsd_z_500 == 0) * (img_data_clsd_z_1000 == 1))

    np.count_nonzero(img_data_500)
    np.count_nonzero(img_data_clsd_z_500)

    imgfinale = nib.Nifti1Image(img_data_clsd_z_1000_superposable * 1000 + img_data_clsd_z_500 * 500 + img_data_clsd_z_1_superposable, img.affine)

    back_T2_data = back_T2.get_data()
    back_T2_data = back_T2_data * (img_data_clsd_z_1000 == 0)
    back_T1_data = back_T1.get_data()
    back_T1_data = back_T1_data * (img_data_clsd_z_1 == 0)

    img_data_finale_avec_back_T2 = back_T2_data + img_data_clsd_z_1000
    img_data_finale_avec_back_T1 = back_T1_data + img_data_clsd_z_1

    img_finale_avec_back_T1 = nib.Nifti1Image(img_data_finale_avec_back_T1, img.affine)
    img_finale_avec_back_T2 = nib.Nifti1Image(img_data_finale_avec_back_T2, img.affine)

    #img_finale_avec_back_T1.to_filename(os.path.join(os.path.join("/home/hs252699/test_fermeture",id_patient,"model04"), (id_patient + '_enh-gado_T1w_bfc_clsd.nii.gz')))
    #img_finale_avec_back_T2.to_filename(os.path.join(os.path.join("/home/hs252699/test_fermeture",id_patient,"model06"), (id_patient + '_rAxT2_clsd.nii.gz')))
    #imgfinale.to_filename(os.path.join(os.path.join("/home/hs252699/test_fermeture",id_patient,"model05"), (id_patient + '_enh-gado_T1w_bfc_WS_hybrid_voxels_clsd.nii.gz'))

    img_finale_avec_back_T1.to_filename(os.path.join(os.path.join(base_directory, id_patient, "model04"), (id_patient + '_enh-gado_T1w_bfc_clsd.nii.gz')))
    img_finale_avec_back_T2.to_filename(os.path.join(os.path.join(base_directory, id_patient, "model06"), (id_patient + '_rAxT2_clsd.nii.gz')))
    imgfinale.to_filename(os.path.join(os.path.split(filename_mask)[0], (id_patient + '_enh-gado_T1w_bfc_WS_hybrid_voxels_clsd.nii.gz')))



m = glob.glob("/neurospin/radiomics/studies/metastasis/base//[0-9]*[0-9]//model05//*hybrid_voxels.nii.gz")
i1 = glob.glob("/neurospin/radiomics/studies/metastasis/base//[0-9]*[0-9]//model02//*bfc.nii.gz")
i2 = glob.glob("/neurospin/radiomics/studies/metastasis/base//[0-9]*[0-9]//model01//*rAxT2.nii.gz")

for i in range(m.__len__()):
    creation_mask(m[i], i1[i], i2[i], "disk")
