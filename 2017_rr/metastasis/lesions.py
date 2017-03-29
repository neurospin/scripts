# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:50:13 2017

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import nibabel as ni
import numpy as np
from numpy import unique
from glob import glob
import os.path
import argparse
import re

tissues = ['edema', 'enh', 'necrosis']
models = ['model10', 'model11']
path='/neurospin/radiomics/studies/metastasis/base'
fenh = glob(os.path.join(path, '*'))
sid = [i.split('/')[6] for i in fenh]
sid = unique(sid).tolist()[0:-5]

doc = """
This command to help reorganize manually segmented data
"""
parser = argparse.ArgumentParser(description=doc)
parser.add_argument(
    "-p", "--process", dest="process", action='store_true',
    help="if activated execute.")


args = parser.parse_args()
for s in sid :
    print s
    for model in models :
        print model
        masks = {}
        all_les = glob('{}/{}/{}/*_mask_[edema|enh|necrosis]*'.format(path, s, model))
        n=0
        for les in all_les :
            num=re.search(r"_([0-9][0-9]?)\.", les)
            nb=re.search(r"[0-9]+", num.group(0))
            no=int(nb.group(0))
            if no > n :
                n=no
        lesions=[1]
        if n > 1 :
            lesions = range(1, n+1)
        for l in lesions:
            #print l
            for t in tissues:
                #print t
                p = '{}/{}/{}/*_mask_{}_{}.nii.gz'.format(path, s, model, t, l)
                p = glob(p)
                #print p
                if len(p) == 1:
                    p = p[0]
                    if not l in masks:
                        masks[l] = {}
                    masks[l][t] = ni.load(p)
        
        for lesion in masks:
            for tissue in tissues:
                if tissue in masks[lesion]:
                    print lesion, "-", tissue, ": ", np.unique(masks[lesion][tissue].get_data())
                    
        if args.process:
            for lesi in masks:
                #print(lesi)
                for tiss in masks[lesi]:
                    #print(tiss)
                    #print(masks[lesi])
                    #if tiss in masks[lesi]:
                    #print(tiss)
                    #print(lesi)
                    ttype = masks[lesi][tiss].get_data_dtype()
                    #print(ttype)
                    tshape = list(masks[lesi][tiss].header['dim'][1:4])+[len(masks[lesi])]
                    #print(tshape)
                    taffine = masks[lesi][tiss].affine
                    #print(taffine)
                    break;
                buf = np.zeros(tshape, dtype=ttype)
                # now create dyn image file to store all subtypes in one lesion file        
                i = 0
                for tissu in tissues:
                    if tissu in masks[lesi]:
                        print(tissu)
                        print(lesi)
                        d=masks[lesi][tissu].get_data()
                        if tissu == 'edema':
                            d[d!=0] = 1
                        else : 
                            if tissu == 'enh':
                                d[d!=0] = 2
                            else :
                                if tissu == 'necrosis':
                                    d[d!=0] = 3
                        print "---> lesion ", lesi, "-", tissu, ": ", np.unique(d)
                        buf[:,:,:,i] = d
                        i += 1
                
#                flesion = os.path.basename(masks[lesion][tissue].get_filename()).split('_')[0]
#                mlesion = os.path.basename(masks[lesion][tissue].get_filename()).split('_')[1]
                flesion = '{}_{}_mask_lesion-{}'.format(s, model, lesi) + '.nii.gz'
                f=os.path.join(path, s, model, flesion)
                print f
                ni.save(ni.Nifti1Image(buf, affine=taffine), f)