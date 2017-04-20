#!/usr/bin/env python
import argparse
import sys
import os.path
import subprocess
import nibabel as ni
import traceback
import shutil
import tempfile
from glob import glob
from nilearn import plotting
from numpy import unique


MNI_BRAIN = '/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'

def is_dir(dirarg):
    """ Type for argparse - checks that output dir exists.
    """
    if not os.path.isdir(dirarg):
        raise argparse.ArgumentError(
            "The dir '{0}' does not exist!".format(dirarg))
    return dirarg

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', metavar='FILE', required=True,
                    help='Image corrected for bias field')
parser.add_argument('-d', '--outdir', metavar='PATH', type=is_dir,
                    required=True,
                    help='Output directory to create the output files.')
parser.add_argument(
    "-r", "--rescue", dest="rescue", action='store_true',
    help="if activated activate the rescue mode.")

def rescue(args):
    # Initialize extractor
    try:

        # get an tmp dir change dir
        tmpdir = tempfile.mkdtemp()
        prevdir = os.getcwd()
        os.chdir(tmpdir)
        
        # 
        ImageFilePath = (args.image).replace('model02', 'model03')
        OutDirPath = args.outdir
        #
        atlas_im = '/neurospin/radiomics/studies/metastasis/base/187962757123/model02/187962757123_enh-gado_T1w_bfc.nii.gz'
        atlas_nat2mni = '/neurospin/radiomics/studies/metastasis/base/187962757123/model03/nat2mni'

        # intermediate path
        cmd = ['fsl5.0-flirt',
               '-in', ImageFilePath,
               '-ref', atlas_im,
               '-dof', '12',
               '-omat', 'image2atlas']
        print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
        results = subprocess.check_call(cmd)   
        
        cmd = ['fsl5.0-convert_xfm',
               '-omat', 'nat2mni',
               '-concat', atlas_nat2mni, 'image2atlas']
        print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
        results = subprocess.check_call(cmd)
        
        #invert Xformation
        cmd = [
            'fsl5.0-convert_xfm',
            '-omat', 'mni2nat',
            '-inverse', 'nat2mni'
            ]
        print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
        results = subprocess.check_call(cmd)

        cmd = ['fsl5.0-flirt',
               '-applyxfm',
               '-init', 'mni2nat',
               '-ref', ImageFilePath,
               '-in', os.path.join(os.path.dirname(ImageFilePath), 'hatbox.nii.gz'),
               '-out', 'native_hatbox.nii.gz'
               ]
        print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
        results = subprocess.check_call(cmd)
        hat_box = ni.load('native_hatbox.nii.gz')
        hat_data = hat_box.get_data()
        hat_data[hat_data != 0] = 1
        ni.save(ni.Nifti1Image(hat_data, affine=hat_box.get_affine()),
                'native_hatbox.nii.gz')

        # create QC for hat_box steps
        bg = ni.load('native_hatbox.nii.gz')
        display = plotting.plot_anat(
                    ni.load(ImageFilePath), 
                    title="T1 Gado with hatbox contours",
                    display_mode = 'z',
                    cut_coords = 10)
        display.add_overlay(bg, threshold=0)
        display.savefig("QC_axial_hatbox.pdf")
        display.close()
        display = plotting.plot_anat(
                    ni.load(ImageFilePath), 
                    title="T1 Gado with hatbox contours",
                    display_mode = 'x',
                    cut_coords = 10)
        display.add_overlay(bg, threshold=0)
        display.savefig("QC_sagital_hatbox.pdf")
        display.close()

        # move selected files
        flist = ['image2atlas', 'mni2nat',
                 'nat2mni', 'native_hatbox.nii.gz', 'QC_axial_hatbox.pdf',
                 'QC_sagital_hatbox.pdf']
        for f in flist:
            if os.path.exists(os.path.join(OutDirPath, f)):
                os.remove(os.path.join(OutDirPath, f))
        for f in flist:
            shutil.move(f, OutDirPath)
        
    except Exception:
        print 'model03_mni rescue registration/segmentation FAILED:\n%s', traceback.format_exc()

    # finale housekeeping
    os.chdir(prevdir)
    shutil.rmtree(tmpdir)

 
def regular(args):
    # Initialize extractor
    if glob(os.path.join(args.outdir, '*')) != []:
        raise Exception("The dir '{0}' is not empty!".format(args.outdir))

    try:

        # get an tmp dir change dir
        tmpdir = tempfile.mkdtemp()
        prevdir = os.getcwd()
        os.chdir(tmpdir)

        # 
        ImageFilePath = args.image
        work_in = os.path.basename(ImageFilePath) 
        OutDirPath = args.outdir

        # base/XX/model02/XX_bfc.nii.gz -> cropped
        tmp_cropped = 'fov_cropped'
        # create cropped volume
        cmd = ['fsl5.0-robustfov',
               '-i', ImageFilePath,
               '-m', tmp_cropped]
        print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
        results = subprocess.check_call(cmd)

        # get cropped xfromation
        cmd = ['fsl5.0-robustfov',
               '-i', ImageFilePath,
               '-r', tmp_cropped]
        print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
        results = subprocess.check_call(cmd)

        # apply xfromation to get the file cropped in working dir and remain
        # in native referential
        cmd = ['fsl5.0-flirt',
               '-in', tmp_cropped,
               '-ref', ImageFilePath, 
               '-init', tmp_cropped, 
               '-applyxfm',
               '-out', work_in ]
        print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
        results = subprocess.check_call(cmd)
            
        # perform bet extraction and mask transcoding in float32
        mask_bet = work_in.replace('bfc', 'bfc_betmask')
        mask_bin = mask_bet.replace('betmask', 'betmask_mask')
        cmd = ['fsl5.0-bet', work_in, mask_bet, '-m', '-S', '-R']
        print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
        results = subprocess.check_call(cmd)
        
        print ">>> ", os.getcwd(), " <<<", "Convert bet mask in float image: ",mask_bin
        nimg = ni.load(mask_bin)
        ni.save(ni.Nifti1Image(nimg.get_data().astype('float32'),
                               affine=nimg.get_affine()),
                mask_bin)

        # performe fast segmentation
        # --out flag does not seem to work : ugly workaround ...
        cmd = ['fsl5.0-fast', 
               ' -t 1 -n 3 ', 
               mask_bet]
        print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
        results = subprocess.check_call(cmd)

#        rename_fns = unique(glob('*seg*') + glob('*pve*')).tolist()
#        for rename_fn in rename_fns:
#            shutil.move(rename_fn, 
#                        rename_fn.replace('betmask', 
#                                          work_in.replace('.nii.gz', '')))
#        rename_fns = unique(glob('*betmask*')).tolist()
#        for rename_fn in rename_fns:
#            shutil.move(rename_fn,
#                        rename_fn.replace('betmask', 
#                                          '{}_betmask'.format(work_in.replace('.nii.gz', '')))
        # QC :  pdf sheet
        bg = ni.load(work_in.replace('.nii.gz', '_betmask_pveseg.nii.gz'))
        # image axial
        display = plotting.plot_anat(bg, title="FAST segmentation", 
                                     display_mode = 'z',
                                     cut_coords = 20)
        display.savefig('{}_pveseg.pdf'.format(work_in.replace('.nii.gz', '')))
        display.close()

        
        # affine registration -in betmask -ref MNI_brain_1mm
        cmd = [
            'fsl5.0-flirt',
            '-dof', '12',
            '-omat', 'nat2mni',
            '-in', mask_bet,
            '-ref', MNI_BRAIN
            ]
        print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
        results = subprocess.check_call(cmd)

        #invert Xformation
        cmd = [
            'fsl5.0-convert_xfm',
            '-omat', 'mni2nat',
            '-inverse', 'nat2mni'
            ]
        print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
        results = subprocess.check_call(cmd)
        tmp_cropped = 'fov_cropped'

        # create a 40 mm height hat box in mni
        mni = ni.load(MNI_BRAIN)
        hat_data = mni.get_data()
        hat_data[hat_data < 800] = 0
        _, _, zmax = hat_data.shape
        hat_data[:, :, 121:zmax] = 0
        hat_data[:, :, 0:79] = 0
        hat_data[hat_data != 0] = 1
        ni.save(ni.Nifti1Image(hat_data, affine=mni.get_affine()),
                'hatbox.nii.gz')

        #
        cmd = ['fsl5.0-flirt',
               '-applyxfm',
               '-init', 'mni2nat',
               '-ref', ImageFilePath,
               '-in', 'hatbox.nii.gz',
               '-out', 'native_hatbox.nii.gz'
               ]
        print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
        results = subprocess.check_call(cmd)
        hat_box = ni.load('native_hatbox.nii.gz')
        hat_data = hat_box.get_data()
        hat_data[hat_data != 0] = 1
        ni.save(ni.Nifti1Image(hat_data, affine=hat_box.get_affine()),
                'native_hatbox.nii.gz')

        # create QC for hat_box steps
        bg = ni.load('native_hatbox.nii.gz')
        display = plotting.plot_anat(
                    ni.load(ImageFilePath), 
                    title="T1 Gado with hatbox contours",
                    display_mode = 'z',
                    cut_coords = 10)
        display.add_overlay(bg, threshold=0)
        display.savefig("QC_axial_hatbox.pdf")
        display.close()
        display = plotting.plot_anat(
                    ni.load(ImageFilePath), 
                    title="T1 Gado with hatbox contours",
                    display_mode = 'x',
                    cut_coords = 10)
        display.add_overlay(bg, threshold=0)
        display.savefig("QC_sagital_hatbox.pdf")
        display.close()

        # move selected files
        flist = unique(glob('*'))
        for f in flist:
            shutil.move(f, OutDirPath)

    except Exception:
        print 'model03_mni registration/segmentation FAILED:\n%s', traceback.format_exc()

    # finale housekeeping
    os.chdir(prevdir)
    shutil.rmtree(tmpdir)


#
#
#if __name__ == "__main__":
#    print ">>>>>>>>>>>>>> ###############"

args = parser.parse_args()
if args.rescue:
    rescue(args)
else:
    regular(args)
