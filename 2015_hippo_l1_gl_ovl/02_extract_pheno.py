#! /usr/bin/env python
##########################################################################
# Brainomics - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy
import pickle
import os
import optparse
import subprocess
from glob import glob

#
base = dict(Hippocampus_L=dict(l="16.5", u="17.5"),
                 Hippocampus_R=dict(l="52.5", u="53.5"),)



if __name__ == "__main__":
    structure = 'Hippocampus_L'
    pin = ('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data'
        'sts_asym_rightonly.phe')
    pout = ('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/')
    pin = ('/neurospin/imagen/processed/fslfirst')

    parser = optparse.OptionParser()
    parser.add_option('-s', '--struct',
                      help='struct prefix',dest="structure",
                      default=structure, type="string")
    parser.add_option('-i', '--path',
                      help='path input DB', dest="pin",
                      default=pin, type="string")
    parser.add_option('-o', '--out',
                      help='path output file', dest="pout",
                      default=pout, type="string")

    (options, args) = parser.parse_args()
    print options.structure
    fout = os.path.join(options.pout, options.structure+'.csv')
    print "will create: ", fout
    #
    fin = glob(os.path.join(options.pin, '*', '*firstseg.nii.gz'))
    res = []
    for name in fin:  
        subject = name.split('/')[-2]
        print 'Processing subject: ', subject
        cmd = 'fslstats %s -l %s -u %s -V' % (name, base[options.structure]['l'],
                                              base[options.structure]['u'])
        tmp  = subprocess.check_output(cmd, shell=True)
        print tmp.split()
        res.append('%s\t%s\t%s'%(subject,subject,tmp.split()[1]))
    
    fp = open(fout,"w")
    fp.write('FID\tIID\t%s\n'%options.structure)
    fp.write("\n".join(res))
    fp.write('\n')
    fp.close()
    
"""
cd /neurospin/imagen/processed/fslfirst/000074104786

fsl5.0-applywarp -i /usr/share/data/harvard-oxford-atlases/HarvardOxford/HarvardOxford-sub-prob-1mm.nii.gz -r 000074104786_all_fast_firstseg.nii.gz -o /tmp/HarvardOxford-sub-prob-1mm.nii.gz 

"""