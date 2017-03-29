# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:30:30 2016

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
from nilearn import plotting
import nibabel
import argparse
import os
from clindmri.registration.fsl import flirt
from glob import glob

from hopla.converter import hopla

# Parameters to keep trace
__hopla__ = ["tool", "version"]

doc = """
Hopla context to invoke register_pitie.py. Perform unit registration of 
T2 on T1 image using fsl/flirt command. Use mutual info and trigger sinc resampling

Organize results according the project rules.

Command:
========

python hl_register_pitie.py \
        -b /volatile/frouin/radiomique_radiogenomique \
        -r AxT1enhanced  \
        -i AxT2
        

python hl_register_pitie.py \
        -b /volatile/frouin/radiomique_radiogenomique \
        -r AxT1enhanced  \
        -i AxT2 \
        -p

"""

# Parsing 
def is_dir(dirarg):
    """ Type for argparse - checks that output dir exists.
    """
    if not os.path.isdir(dirarg):
        raise argparse.ArgumentError(
            "The dir '{0}' does not exist!".format(dirarg))
    return dirarg

parser = argparse.ArgumentParser(description=doc)
parser.add_argument(
    "-v", "--verbose", dest="verbose", type=int, choices=[0, 1, 2], default=0,
    help="increase the verbosity level: 0 silent, [1, 2] verbose.")
parser.add_argument(
    "-b", "--basedir", dest="basedir", required=True, metavar="PATH",
    help="the subject main directory to parse files",
    type=is_dir)
parser.add_argument(
    "-r", "--ref", dest="refile", metavar="STRING",
    help="Motif (regular expression)  to serarch for ref images")
parser.add_argument(
    "-s", "--subject", dest="subject", metavar="STRING",
    help="An existing subject in the data tree")
parser.add_argument(
    "-i", "--in", dest="infile", metavar="STRING",
    help="Motif (regular expression)  to serarch for images to register")
parser.add_argument(
    "-p", "--process", dest="process", action='store_true',
    help="if activated execute.")
args = parser.parse_args()


# glob the file to analyze
# list subjects from basedir
#print glob(os.path.join(args.basedir, '*', args.infile))

if args.subject is not None:
    infiles = set([os.path.dirname(os.path.abspath(i)) for i in \
                         glob(os.path.join(args.basedir, '{}'.format(args.subject), args.infile))])
    refiles = set([os.path.dirname(os.path.abspath(i)) for i in  \
                         glob(os.path.join(args.basedir, '{}'.format(args.subject), args.refile))])
else:
    infiles = set([os.path.dirname(os.path.abspath(i)) for i in \
                             glob(os.path.join(args.basedir, '*', args.infile))])
    refiles = set([os.path.dirname(os.path.abspath(i)) for i in  \
                             glob(os.path.join(args.basedir, '*', args.refile))])
subjects = set.intersection(infiles, refiles)
subjects = [os.path.basename(i) for i in subjects]
# get infiles and filter them
infiles = []
refiles =  []
for s in subjects:
    infiles.extend(glob(os.path.join(args.basedir, '{}'.format(s),
                            args.infile, '{}.nii.gz'.format(args.infile))))
    refiles.extend(glob(os.path.join(args.basedir, '{}'.format(s),
                            args.refile, '{}.nii.gz'.format(args.refile))))
outfiles = [i.replace(args.basedir, os.path.join(args.basedir, 'preprocess')) for i in infiles]
outfiles = [i.replace(args.infile,'',1) for i in outfiles]
outfiles = [os.path.splitext(os.path.splitext(i)[0])[0] for i in outfiles]
outfiles = [os.path.join('{}'.format(os.path.dirname(i)),
                        'r{}.nii.gz'.format(os.path.basename(i)))
                        for i in outfiles]

if args.process:
    logfile = "{}/preprocess/log.txt".format(args.basedir)
    if not os.path.isdir(os.path.dirname(logfile)):
        os.makedirs(os.path.dirname(logfile))
    #
    status, exitcodes = hopla(
        os.path.join('/volatile','frouin','radiomique_radiogenomique',
                     'register_pitie.py'),
        i=infiles,
        r=refiles,
        o=outfiles,
        hopla_iterative_kwargs=["i", "r", "o"],
        hopla_cpus=2,
        hopla_logfile=logfile,
        hopla_verbose=1)
