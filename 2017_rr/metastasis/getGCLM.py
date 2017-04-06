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

import matplotlib.pyplot as plt
from radiomics.featureextractor import RadiomicsFeaturesExtractor


doc = """
source /volatile/frouin/pyrad/bin/activate
python $HOME/gits/scripts/2017_rr/metastasis/getGCLM.py \
   --param $HOME/gits/scripts/2017_rr/metastasis/minimal.yaml \
   --out /tmp/187962757123_glcm.json \
   --format json \
   /neurospin/radiomics/studies/metastasis/base/187962757123/model03/187962757123_enh-gado_T1w_bfc_WS.nii.gz \
   $HOME/gits/scripts/2017_rr/metastasis/187962757123_enh-gado_T1w_bfc_WS_rad-ttype-lesion-1.nii.gz
"""

parser = argparse.ArgumentParser()
parser.add_argument('image', metavar='Image',
                    help='Features are extracted from the Region Of Interest '
                         '(ROI) in the image')
parser.add_argument('mask', metavar='Mask',
                    help='Mask identifying the ROI in the Image')

parser.add_argument('--out', '-o', metavar='FILE', nargs='?',
                    type=argparse.FileType('w'), default=sys.stdout,
                    help='File to append output to')
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

                # subplots: one histogram + co-occurences matrices
                nb_coocc_matrices = featureClass.P_glcm.shape[2]
                nb_subplots = 1 + nb_coocc_matrices
                fig, axes = plt.subplots(nrows=1, ncols=nb_subplots,
                                         figsize=(18, 2))
                histo_ax, matrices_axes = axes[0], axes[1:]
                fig.suptitle("GLCM matrices, image type: %s, bin width: %i"
                             % (inputImageName, featureClass.binWidth))

                # Histogram
                bins = featureClass.binEdges
                histo_ax.hist(featureClass.matrix.flatten(), bins=bins)
                histo_ax.tick_params(labelsize=3)
                histo_ax.set_title("%s hist" % inputImageName, fontsize=8)

                # Identify global min/max of concurrent matrices to have a
                # consistent coloration across all images
                co_min = featureClass.P_glcm.min()
                co_max = featureClass.P_glcm.max()

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
                name_png = '%s_%s_bw%s.png' % (featureClassName,
                                               inputImageName,
                                               featureClass.binWidth)
                path_png = os.path.join(outdir, name_png)
                plt.savefig(path_png, dpi=300)

        return featureVector


def main():
    args = parser.parse_args()

    global outdir  # Make outdir visible to the computeFeatures method
    # Save PNG snapshots in the same directory as the output features
    outdir = os.path.dirname(args.out.name)

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

    # Initialize extractor
    try:
        if args.param is not None:
            extractor = ModifiedRadiomicsFeaturesExtractor(args.param[0])
        else:
            extractor = ModifiedRadiomicsFeaturesExtractor()
        logging.info('Extracting features with kwarg settings: '
                     '%s\n\tImage:%s\n\tMask:%s',
                     str(extractor.kwargs), os.path.abspath(args.image),
                     os.path.abspath(args.mask))
        featureVector = collections.OrderedDict()
        featureVector['image'] = os.path.basename(args.image)
        featureVector['mask'] = os.path.basename(args.mask)

        featureVector.update(extractor.execute(args.image, args.mask,
                                               args.label))

        if args.format == 'csv':
            writer = csv.writer(args.out, lineterminator='\n')
            writer.writerow(featureVector.keys())
            writer.writerow(featureVector.values())
        elif args.format == 'json':
            json.dump(featureVector, args.out, indent=4)
            args.out.write('\n')
        else:
            for k, v in featureVector.iteritems():
                args.out.write('%s: %s\n' % (k, v))
    except Exception:
        logging.error('FEATURE EXTRACTION FAILED:\n%s', traceback.format_exc())

    args.out.close()
    args.log_file.close()


if __name__ == "__main__":
    main()
