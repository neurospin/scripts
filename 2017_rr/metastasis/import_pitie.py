#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2013 - 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from __future__ import print_function
import argparse
import os
from zipfile import ZipFile
from numpy import unique
from shutil import rmtree
from glob import glob


# Bredala import
try:
    import bredala
    bredala.USE_PROFILER = False
    #bredala.register("pydcmio.dcmreader.reader",
    #                 names=["get_values"])
except:
    pass

# Dcmio import
from pydcmio import __version__ as version
from pydcmio.dcmreader.reader import walk
from pydcmio.dcmreader.reader import get_values
from pydcmio.dcmreader.reader import STANDARD_EXTRACTOR

from hopla.converter import hopla

# Parameters to keep trace
__hopla__ = ["tool", "version"]


SEQ_NAME_MAP ={
    'Ax FSPGR G':               'AxT1enhanced',
    'Ax FSPGR 1.2 mm 1 plan':   'AxT1enhanced',
    'Ax FSPGR 3D 1.2mm':        'AxT1enhanced',
    'Ax FSPGR 3D HD 1.2mm':     'AxT1enhanced',
    'AX CUBE T1':               'AxT1enhanced',
    '3D GADO':                  'AxT1enhanced',
    '3D T1 BRAVO gado':         'AxT1enhanced',
    'Ax T2* GRE' :              'AxT2',
    'AxFLAIR GADO' :            'AxT2',
    'FLAIR' :                   'AxT2',
    'AxFLAIR' :                 'AxT2',
    'Ax FLAIR' :                'AxT2',
    'Ax T2 FLAIR 5/0' :         'AxT2'}
SEQ_NAME = SEQ_NAME_MAP.keys()

# Script documentation
doc = """
Dicom parser for Pitie NeuroRadio Import
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This code fix and list the file to be processed to import NeuroRadio onco 
images.

Based on Dicom field sequences that are parsed recursively (deep search) 
to handle enhanced storage parsing.

Steps:



Command:

python import_pitie.py \
    -i source/ -o raw/  -b base

"""


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
    "-p", "--process", dest="process", action='store_true',
    help="if activated execute.")
parser.add_argument(
    "-d", "--dryrun", dest="dryrun", action='store_true',
    help="dryrun : print seqnames and corresponding directory pathes. No conversion run")
parser.set_defaults(dryrun=False)
parser.add_argument(
    "-l", "--logfile", dest="logfile", metavar="FILE",
    help="logfile filename")
parser.add_argument(
    "-i", "--indirectory", dest="dcmdir", required=True, metavar="PATH",
    help="the subject main directory to parse for DICOM zip files",
    type=is_dir)
parser.add_argument(
    "-o", "--outdirectory", dest="outdir", required=True, metavar="PATH",
    help="the outdir that contains directory ready for dcmio_XXX cmds",
    type=is_dir)
parser.add_argument(
    "-b", "--basedirectory", dest="bdir", required=True, metavar="PATH",
    help="the final dir entry thath will contain proper converted dicom files",
    type=is_dir)
args = parser.parse_args()


log_msg = []

"""
Welcome message
"""
if args.verbose > 0:
    print("[info] Start dicom parsing...")
    print("[info] Zip dir: {0}.".format(args.dcmdir))
    print("[info] Dicom dir: {0}.".format(args.outdir))


"""
Parse inDirectory with the zip files
"""
ziplist = glob(os.path.join(args.dcmdir, '*.zip'))
unziplist = []
for z in ziplist:
    print('Unzipping {}...'.format(z))
    with ZipFile(z,"r") as z_fp:
        subject_dirname = (os.path.basename(z).split('.')[0]).split('_')[1]
        unzdir =  os.path.join(args.outdir, subject_dirname)
        os.mkdir(unzdir)
        z_fp.extractall(unzdir)
        unziplist.append(unzdir)

"""
Fixing the unzipped dirs
"""
# remove VERSION file
for unzdir in unziplist:
    for root, dirs, files in os.walk(unzdir):
        for name in files:
            if 'VERSION' in name:
                os.remove(os.path.join(root, name))

# remove non dicom dir and get parameters
unzargs = {}
for unzdir in unziplist:
    if args.dryrun:
        log_msg.append('=======================seq_name============================')
    sequences = []
    sequences_path = []
    to_del = []
    for root, dirs, files in os.walk(unzdir):
        for name in dirs:        
            # must be a leaf directory
            if  any([os.path.isdir(i)
                     for i in glob(os.path.join(root, name,'*'))]):
                 continue
            # assume at least one file per dir.
            dcmfile = glob(os.path.join(root, name,'*'))[0]
            # dir with jpeg content
            if '.jpg' in dcmfile:
                to_del.append(os.path.join(root, name))
            # supposedly pure dicom content directory
            #     dir with dicom non compliant files are discarded
            else:
                try:
                    seqn = get_values(dcmfile, 'get_sequence_name')
                    seqm = get_values(dcmfile, 'get_manufacturer_name')
#                    seqs = get_values(dcmfile, 'get_nb_slices')                
                except ValueError as e:
                    to_del.append(os.path.join(root, name))
                if args.dryrun:
                    try:
                        seqs = get_values(dcmfile, 'get_nb_slices')
                        log_msg.append('>>> seq_name: {0}\t\t[{1}] #slice:{2} {3}'.format(
                                        seqn, os.path.join(root, name),
                                        seqs,seqm))
                    except Exception as e: 
                        log_msg.append('>>> seq_name: {0}\t\t[{1}] #slice:{2} {3}'.format(
                                        seqn, os.path.join(root, name),
                                        -1, seqm))
                        continue
                else: 
                    if seqn not in SEQ_NAME:
                        to_del.append(os.path.join(root, name))
                    else:
                        sequences_path.append(os.path.join(root, name))
                        sequences.append(seqn)
    #now do the deletion job
    for d in to_del:
        rmtree(d)
    # update information for each dir
    unzargs[os.path.basename(unzdir)] = {'path':os.path.join(unzdir), 
                                         'sequences':sequences, 
                                         'sequences_path':sequences_path} 

"""
Prepare the path'es
"""
# To perform a pydcmio_splitseries
splitparameter = {}
for subject in unzargs:
    splitparameter[subject] = {}
    splitparameter[subject]['inpath'] = []    
    splitparameter[subject]['outpath'] = []
    #
    for sp in unique([os.path.dirname(i) for i in unzargs[subject]['sequences_path']]):
        #print(sp)
        inpath = sp
        outpath = os.path.join(unzargs[subject]['path'], 'splitted')
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        splitparameter[subject]['inpath'].append(inpath)
        splitparameter[subject]['outpath'].append(outpath)

# to perform a pydcmio_dicom2nifti
dcm2niiparameter = {}
for subject in unzargs:
    dcm2niiparameter[subject] = {}
    dcm2niiparameter[subject]['inpath'] = []    
    dcm2niiparameter[subject]['outpath'] = []
    dcm2niiparameter[subject]['proto'] = []
    #
    for seq, seqp in zip(unzargs[subject]['sequences'], unzargs[subject]['sequences_path']):
        outpath = os.path.join(args.bdir)
        inpath = seqp
        proto = SEQ_NAME_MAP[seq]
        #
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        #
        dcm2niiparameter[subject]['inpath'].append(inpath)
        dcm2niiparameter[subject]['outpath'].append(outpath)
        dcm2niiparameter[subject]['proto'].append(proto)

print('===== To perform pydcmio_splitseries =================================')
for s in splitparameter:
    for i, o in zip(splitparameter[s]['inpath'],
                       splitparameter[s]['outpath']):
        print('pydcmio_splitseries -v 2 -i {0} -o {1}'.format(i, o))

print('===== To perform pydcmio_dicom2nifti =================================')
sids = []
protocols = []
dcmdirs = []
for s in dcm2niiparameter:
    for i, o, p in zip(dcm2niiparameter[s]['inpath'],
                       dcm2niiparameter[s]['outpath'], 
                       dcm2niiparameter[s]['proto']):
        print('pydcmio_dicom2nifti -v 2 -d {0} -o {1} -s {2} -p {3}'.format(
                i,o,s,p))
        sids.append(s)
        protocols.append(p)
        dcmdirs.append(i)

if args.logfile is not None:
    with open(args.logfile, 'w') as fp:
        fp.write('\n'.join(log_msg))
        fp.write('\n')
else:
    print('\n'.join(log_msg))



if args.process:
    logfile = "/volatile/frouin/radiomique_radiogenomique/base2/log.txt"
    #
    sids = []
    protocols = []
    dcmdirs = []
    for s in dcm2niiparameter:
        for i, o, p in zip(dcm2niiparameter[s]['inpath'],
                           dcm2niiparameter[s]['outpath'], 
                           dcm2niiparameter[s]['proto']):
            sids.append(s)
            protocols.append(p)
            dcmdirs.append(os.path.join('/volatile/frouin/radiomique_radiogenomique/',i))
    #
    status, exitcodes = hopla(
        os.path.join(os.getenv('HOME'), "gits", "pydcmio", "pydcmio",
                     "scripts", "pydcmio_dicom2nifti"),
        s=sids,
        p=protocols,
        d=dcmdirs,
        o="/volatile/frouin/radiomique_radiogenomique/base2",
        t=True,
        r="/volatile/frouin/radiomique_radiogenomique/base2/transcoding.json",
        x=True,
        hopla_iterative_kwargs=["s", "p", "d"],
        hopla_cpus=2,
        hopla_logfile=logfile,
        hopla_verbose=1)


