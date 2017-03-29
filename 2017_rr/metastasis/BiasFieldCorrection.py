# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:16:46 2017

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
#!/usr/bin/env python

import argparse
import sys
import os.path
import logging
import traceback
import csv
import json
import subprocess
import nibabel as ni

# Script documentation
doc = """
Bias field correction
~~~~~~~~~~~~~~~~~~~~~

This code uses N4 (ITK4.7 wrapped into ANTs) code to correct for BiasField
A BiasField map is generated (see _bias.nii.gz file)

Description: Performs image bias correction using N4 algorithm. This module is based on the ITK filters
contributed in the following publication: Tustison N, Gee J “N4ITK: Nick’s N3 ITK Implementation For MRI
Bias Field Correction”, The Insight Journal 2009 January-June, http://hdl.handle.net/10380/3053
version: 9
documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/N4ITKBiasFieldCorrection
contributor: Nick Tustison (UPenn), Andrey Fedorov (SPL, BWH), Ron Kikinis (SPL, BWH)
acknowledgements: The development of this module was partially supported by NIH grants R01 AA016748-01,
R01 CA111288 and U01 CA151261 as well as by NA-MIC, NAC, NCIGT and the Slicer community.

Steps:


Command:
python $HOME/gits/scripts/2017_rr/metastasis/BiasFieldCorrection \
    -i /neurospin/radiomics/studies/metastasis/base/187962757123/anat/187962757123_enh-gado_T1w.nii.gz \
    -o /tmp/187962757123_enh-gado_T1w_bfc.nii.gz

"""


parser = argparse.ArgumentParser()
parser.add_argument('--image', '-i', metavar='FILE',
                    help='Image to correct for bias field')
parser.add_argument('--out', '-o', metavar='FILE',nargs='?',
                    type=argparse.FileType('w'),
                    help='File to append output to')
parser.add_argument('--param', '-p', metavar='FILE', nargs=1, type=str, default=None,
                    help='Parameter file containing the settings to be used in extraction')
parser.add_argument('--logging-level', metavar='LEVEL',
                    choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default='WARNING', help='Set capture level for logging')
parser.add_argument('--log-file', metavar='FILE', nargs='?', type=argparse.FileType('w'), default=sys.stderr,
                    help='File to append logger output to')


"""
Currently Otsu mask is infered from input image.
Might need better mask.

See https://github.com/stnava/ANTsR/blob/master/R/getMask.R

See https://github.com/stnava/ANTsR/blob/master/R/n4BiasFieldCorrection.R
n4BiasFieldCorrection<-function( img , mask=NA, shrinkFactor=4, 
  convergence=list(iters=c(50,50,50,50), tol=0.0000001), 
  splineParam=200, 
  verbose = FALSE)
"""



def main():
    args = parser.parse_args()
#    args = parser.parse_args(['--logging-level', 'DEBUG',
#                              '-o', ['/tmp/img.nii.gz','/tmp/img_bias.nii.gz'],
#                            "/neurospin/radiomics/studies/"
#                            "metastasis/base/187962757123/anat/"
#                            "187962757123_enh-gado_T1w.nii.gz"])
#    

    # Initialize Logging
    logLevel = eval('logging.' + args.logging_level)
    rLogger = logging.getLogger('pyRRI')
    rLogger.handlers = []
    rLogger.setLevel(logLevel)

    logger = logging.getLogger()
    logger.setLevel(logLevel)
    handler = logging.StreamHandler(args.log_file)
    handler.setLevel(logLevel)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    logger.addHandler(handler)

    # Initialize extractor
    try:
        if args.param is not None:
            raise Exception('Reading from param file not implemented yet.')

        ImageFilePath = args.image
        CorrFilePath = args.out
        

        if isinstance(ImageFilePath, basestring) and \
                   os.path.exists(ImageFilePath) and \
                   ImageFilePath.find('nii.gz') != -1:
            ImageFilePath = os.path.abspath(ImageFilePath)
        if CorrFilePath.name.find('nii.gz') != -1:
            BiasFilePath = CorrFilePath.name.replace('.nii.gz', '_bias.nii.gz')
            
        image = ni.load(ImageFilePath)

        n4 = '/neurospin/brainomics/neuroimaging_ressources/ants2.1/N4BiasFieldCorrection'
        cmd = [n4,
               '--bspline-fitting', '[ 300 ]',
               '-d', '3',
               '--input-image', '{}'.format(ImageFilePath),
               '--convergence', '[ 50x50x50x50 ]',
               '--output', '[{},{}]'.format(CorrFilePath.name, BiasFilePath),
               '--shrink-factor', '4']
        print " ".join(cmd)
        results = subprocess.check_call(cmd)

    except Exception:
        logging.error('BIAS FIELD CORRECTION FAILED:\n%s', traceback.format_exc())



if __name__ == "__main__":
  main()

