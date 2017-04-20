#!/usr/bin/env python

import argparse
import sys
import os.path
import logging
import collections
import traceback
import csv
import json
import six
import re
import nibabel as ni
import numpy as np

import SimpleITK
import matplotlib.pyplot as plt
from radiomics.featureextractor import RadiomicsFeaturesExtractor

def get_tissue_meta(roi):
    """ Return a dict read from a jsonfile up in the tree that harbor the
        given roi filename

    Parameters
    ----------
    roi: nifti image (mandatory)
        A nifti image in which contains the sampling ROI.

    Returns
    -------
    retval: dict
        Read from the 'tissuetype.json' found on the filesystem.

    """
    fn = roi.get_filename()
    while fn is not '/':
        fn = os.path.dirname(fn)
        expected_name = os.path.join(fn, 'tissuetype.json')
        if os.path.exists(expected_name):
            with open(expected_name) as fp:
                retval = json.load(fp)
                return retval

    raise Exception('Cannot find a tissuetype.json for metadata completion!')


def label_to_shortname(label, roi):
    """ Return a string by looking up label from the tissuetype.json
        inferred by the get_tissue_roi macro
    """
    dtissue = get_tissue_meta(roi)
    for t in dtissue:
        if dtissue[t]['Index'] == label:
             return t

    raise Exception('Cannot find a tissue {0} from {1}'.format(label, roi.get_filename()))


HABITAT = ['edema', 'enhancement', 'both']


def get_mask_from_lesion(lesion, tag, outdir='.'):
    """ Return a dict read from a jsonfile up in the tree that harbor the
        given roi filename

    Parameters
    ----------
    lesion: lesion filename (mandatory)
        A path to the nifti image in which contains the  ROIs.
    tag: a string in HABITAT

    Returns
    -------
    fmask: Pathname
        Pathname to the desired 3D mask
    fjson: Pathname
        Pathname to the desired json file
    """
    #interpret lesion name and lesion group
    lesion_name = re.search(r"lesion-[0-9][0-9]?", lesion)
    lesion_nb = re.search(r"[0-9]+", lesion_name.group(0))
    subject = os.path.basename(lesion).split('_')[0]
    if not subject.isdigit():
        raise Exception('{}: wrong filename format')
    prefix_outfile = os.path.join(outdir,
                                  '{0}_enh-gado_T1w_bfc_WS_rad-'.format(subject))

    # load the file
    vois = ni.load(lesion)
    cumul = np.zeros(vois.get_shape()[:-1], dtype='int16')
    imgs = ni.four_to_three(vois)
    for t, img3d in enumerate(imgs):
        # get the labels contained in the image should be 0 and [1 or 2 or 3]
        labels = np.unique(img3d.get_data())
        #exactly two labels mandatory
        if labels.shape[0] != 2:
            raise Exception('Cannot find a tissue from tissuetype.json.')
        label = max(labels)
        sname = label_to_shortname(label, vois)
        if not sname in HABITAT:
            continue
        if tag == 'both':
            cumul += img3d.get_data()
        elif tag == sname:  # should be edema or enhancement 
            cumul = img3d.get_data()

    # cumul should contain either edema, enhabcement or both
    fmask = prefix_outfile + 'ttype-{0}{1}.nii.gz'.format(tag, lesion_nb.group(0))
    fout = prefix_outfile + 'ttype-{0}{1}.json'.format(tag, lesion_nb.group(0))
    bin_img3d = np.asarray((cumul > 0) * 1, dtype='uint16')
    ni.save(ni.Nifti1Image(bin_img3d, affine=vois.get_affine()), fmask)

    #
    return fmask, fout


doc = """
source /volatile/frouin/pyrad/bin/activate
python $HOME/gits/scripts/2017_rr/metastasis/getGCLM.py \
   --param $HOME/gits/scripts/2017_rr/metastasis/minimal.yaml \
   --out /tmp/GLCM \
   --format json \
   --habitat both \
   /neurospin/radiomics/studies/metastasis/base/187962757123/model04/187962757123_enh-gado_T1w_bfc_WS.nii.gz \
   /neurospin/radiomics/studies/metastasis/base/187962757123/model10/187962757123_model10_mask_lesion-1.nii.gz
"""

parser = argparse.ArgumentParser()
parser.add_argument('image', metavar='Image',
                    help='Features are extracted from the Region Of Interest '
                         '(ROI) in the image')
parser.add_argument('lesion', metavar='Mask',
                    help='Mask identifying the ROIs in the Image')

parser.add_argument('--out', '-o', metavar='DIR', required=True,
                    help='Directory for output file results')
parser.add_argument('--format', '-f', choices=['txt', 'csv', 'json'],
                    default='txt',
                    help='Format for the output. Default is "txt": one feature'
                    ' per line in format "name:value". For "csv": one row of '
                    'feature names, followed by one row of feature values. '
                    'For "json": Features are written in a JSON format '
                    'dictionary "{name:value}"')
parser.add_argument('--param', '-p', metavar='FILE', nargs=1, type=str,
                    default=None, help='Parameter file containing the settings'
                                       ' to be used in extraction')
parser.add_argument('--label', '-l', metavar='N', nargs=1, default=None,
                    type=int, help='Value of label in mask to use for feature '
                                   'extraction')
parser.add_argument('--habitat', '-a', choices=HABITAT,
                    default='both', help='Habitat to study')
parser.add_argument('--logging-level', metavar='LEVEL',
                    choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR',
                             'CRITICAL'],
                    default='WARNING', help='Set capture level for logging')
parser.add_argument('--log-file', metavar='FILE', nargs='?',
                    type=argparse.FileType('w'), default=sys.stderr,
                    help='File to append logger output to')


class ModifiedRadiomicsFeaturesExtractor(RadiomicsFeaturesExtractor):

    def computeFeatures(self, image, mask, inputImageName, **kwargs):

        #######################################################################
        # Original code

        featureVector = collections.OrderedDict()

        # Calculate feature classes
        for featureClassName, enabledFeatures in six.iteritems(
                self.enabledFeatures):
            # Handle calculation of shape features separately
            if featureClassName == 'shape':
                continue

            if featureClassName in self.getFeatureClassNames():
                self.logger.info('Computing %s', featureClassName)
                featureClass = self.featureClasses[featureClassName](image,
                                                                     mask,
                                                                     **kwargs)

            if enabledFeatures is None or len(enabledFeatures) == 0:
                featureClass.enableAllFeatures()
            else:
                for feature in enabledFeatures:
                    featureClass.enableFeatureByName(feature)

            featureClass.calculateFeatures()
            for (featureName, featureValue) in six.iteritems(
                    featureClass.featureValues):
                newFeatureName = "%s_%s_%s" % (inputImageName,
                                               featureClassName, featureName)
                featureVector[newFeatureName] = featureValue

            ###################################################################
            # Supplementary code to create snapshots for GCLM

            if featureClassName == "glcm":

                # Save ROI, binned ROI and mask as Nifti
                roi_image = SimpleITK.GetImageFromArray(featureClass.imageArray)
                bin_roi_image = SimpleITK.GetImageFromArray(featureClass.matrix)
                # mask_array = SimpleITK.GetImageFromArray(featureClass.maskArray)
                mask_image = featureClass.inputMask
                path_roi = os.path.join(outdir, "%s_roi_%s.nii.gz" % (fn_prefix, featureClassName))
                path_bin_roi = os.path.join(outdir, "%s_binned_roi_%s.nii.gz" %  (fn_prefix, featureClassName))
                path_mask = os.path.join(outdir, "%s_mask_%s.nii.gz" %  (fn_prefix, featureClassName))
                SimpleITK.WriteImage(roi_image, path_roi)
                SimpleITK.WriteImage(bin_roi_image, path_bin_roi)
                SimpleITK.WriteImage(mask_image, path_mask)

                # subplots: one histogram + co-occurences matrices
                nb_coocc_matrices = featureClass.P_glcm.shape[2]
                nb_subplots = 1 + nb_coocc_matrices
                fig, axes = plt.subplots(nrows=1, ncols=nb_subplots,
                                         figsize=(18, 2))
                histo_ax, matrices_axes = axes[0], axes[1:]
                fig.suptitle("GLCM matrices, image type: %s, bin width: %i"
                             % (inputImageName, featureClass.binWidth))

                # Histogram
                #bins = featureClass.binEdges # binEdges are in real data level
                bins = range(1, featureClass.coefficients['Ng']+1)
                # this hist consider all voxels in the bounding box
                # histo_ax.hist(featureClass.matrix.flatten(), bins=bins)
                # this hist consider voxel within the ROI
                histo_ax.hist(
                    featureClass.matrix[np.where(featureClass.maskArray != 0)],
                    bins=bins)
                histo_ax.tick_params(labelsize=3)
                histo_ax.set_title("%s hist" % inputImageName, fontsize=8)

                # Identify global min/max of concurrent matrices to have a
                # consistent coloration across all images
                co_min = featureClass.P_glcm.min()
                co_max = featureClass.P_glcm.max()
#                print(featureClass.P_glcm.shape )

                # Create image subplot for each matrix along with colorbar
                extent = [bins[0], bins[-1], bins[0], bins[-1]]
                for i, ax in enumerate(matrices_axes):
                    co_matrix = featureClass.P_glcm[:, :, i]
                    im = ax.imshow(co_matrix, vmin=co_min, vmax=co_max,
                                   extent=extent, cmap="Reds",
                                   interpolation='nearest')
                    ax.tick_params(labelsize=3)
                    ax.set_title("angle index: %i" % i, fontsize=6)
                    cb = plt.colorbar(im, ax=ax, orientation="horizontal")
                    cb.ax.tick_params(labelsize=3)
                fig.tight_layout()
                name_png = '%s_%s_%s_bw%s.png' % (fn_prefix,
                                               featureClassName,
                                               inputImageName,
                                               featureClass.binWidth)
                path_png = os.path.join(outdir, name_png)
                plt.savefig(path_png, dpi=300)

            if featureClassName == "glrlm":
                nb_coocc_matrices = featureClass.P_glrlm.shape[2]
                nb_subplots = 1 + nb_coocc_matrices
                fig, axes = plt.subplots(nrows=1, ncols=nb_subplots,
                                         figsize=(18, 2))
                histo_ax, matrices_axes = axes[0], axes[1:]
                fig.suptitle("GLCM matrices, image type: %s, bin width: %i"
                             % (inputImageName, featureClass.binWidth))
                # Identify global min/max of concurrent matrices to have a
                # consistent coloration across all images
                co_min = featureClass.P_glrlm.min()
                co_max = featureClass.P_glrlm.max()

                # Create image subplot for each matrix along with colorbar
                extent = [1, featureClass.P_glrlm[:,:,0].shape[1], featureClass.coefficients['Ng'], 1] 
                for i, ax in enumerate(matrices_axes):
                    co_matrix = featureClass.P_glrlm[:, :, i]
                    im = ax.imshow(co_matrix, vmin=co_min, vmax=co_max,
                                   extent=extent, cmap="Reds",
                                   interpolation='nearest')
                    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/10.)
                    ax.tick_params(labelsize=3)
                    ax.set_title("angle index: %i" % i, fontsize=6)
                    cb = plt.colorbar(im, ax=ax, orientation="horizontal")
                    cb.ax.tick_params(labelsize=3)
                fig.tight_layout()
                name_png = '%s_%s_%s_bw%s.png' % (fn_prefix,
                                               featureClassName,
                                               inputImageName,
                                               featureClass.binWidth)
                path_png = os.path.join(outdir, name_png)
                plt.savefig(path_png, dpi=300)
                
        return featureVector


def main():
    args = parser.parse_args()

    global outdir  # Make outdir visible to the computeFeatures method
    global fn_prefix # Make fn_prefix visible to the computeFeatures method
    # Save PNG snapshots in the same directory as the output features
    outdir = args.out

    # Initialize Logging
    logLevel = eval('logging.' + args.logging_level)
    rLogger = logging.getLogger('radiomics')
    rLogger.handlers = []
    rLogger.setLevel(logLevel)

    logger = logging.getLogger()
    logger.setLevel(logLevel)
    handler = logging.StreamHandler(args.log_file)
    handler.setLevel(logLevel)
    handler.setFormatter(logging.Formatter(
        "%(levelname)s:%(name)s: %(message)s"))
    logger.addHandler(handler)

    #Create temporary masks and get their pathname
    fn_mask, fnout = get_mask_from_lesion(args.lesion, args.habitat, outdir)
    fn_prefix = os.path.basename(fnout).split('.')[0]
    fpout = open(fnout, 'w')
#    print fn_mask, fn_jsonout, args.image

    # Initialize extractor
    try:
        if args.param is not None:
            extractor = ModifiedRadiomicsFeaturesExtractor(args.param[0])
        else:
            extractor = ModifiedRadiomicsFeaturesExtractor()
        logging.info('Extracting features with kwarg settings: '
                     '%s\n\tImage:%s\n\tMask:%s',
                     str(extractor.kwargs), os.path.abspath(args.image),
                     os.path.abspath(fn_mask))
        featureVector = collections.OrderedDict()
        featureVector['image'] = os.path.basename(args.image)
        featureVector['mask'] = os.path.basename(fn_mask)

        featureVector.update(extractor.execute(args.image, fn_mask,
                                               args.label))

        if args.format == 'csv':
            writer = csv.writer(fpout, lineterminator='\n')
            writer.writerow(featureVector.keys())
            writer.writerow(featureVector.values())
        elif args.format == 'json':
            json.dump(featureVector, fpout, indent=4)
            fpout.write('\n')
        else:
            for k, v in featureVector.iteritems():
                fpout.write('%s: %s\n' % (k, v))
    except Exception:
        logging.error('FEATURE EXTRACTION FAILED:\n%s', traceback.format_exc())

    fpout.close()
    args.log_file.close()

#
#if __name__ == "__main__":
#
main()
