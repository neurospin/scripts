"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""

import os, glob, re, json, argparse
import pandas as pd
import time

CC_ANALYSIS = False # Soon deprecated usage
sides = ['R', 'L']
sds = {'R' : 'Right', 'L': 'Left'}
dict_h2 = {}
dict_pval = {}
nb = ['.', '-', 'e', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
sds = {'R' : 'Right', 'L': 'Left', 'asym': 'asym'}
sides = ['asym']

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('-f', '--feature', type=str,
                        help="Features to measure")
    parser.add_argument('-s', '--symmetric', type=int,
                        help="Boolean, need to give 0 for False and another int for True " 
                        "Specify if symmetric template is used or not")
    parser.add_argument('-d', '--database', type=str,
                        help='Data base from which we take the cluster')
    parser.add_argument('-t', '--feature_threshold', type=str,
                        help='Specify the feature used for thresholding either sulc or DPF')
    options = parser.parse_args()
    ## INPUTS ###
    feature = options.feature
    SYMMETRIC = bool(int(options.symmetric))
    database_parcel  = options.database

    feature_threshold = options.feature_threshold

    if SYMMETRIC:
        pheno_dir = '/neurospin/brainomics/2016_HCP/new_pheno_threshold_'+feature_threshold+'/pheno_pits_sym_'+feature+'_'+database_parcel+'_Freesurfer_new/'
        working_dir = '/neurospin/brainomics/2016_HCP/new_analysis_threshold_'+feature_threshold+'/pits_analysis_sym_'+feature+'_'+database_parcel+'_Freesurfer_new'
        output = '/neurospin/brainomics/2016_HCP/new_dictionaries_herit_threshold_'+feature_threshold+'/pits_sym_'+feature+'_'+database_parcel+'_Freesurfer_new'
    else:
        pheno_dir = '/neurospin/brainomics/2016_HCP/new_pheno_threshold_'+feature_threshold+'/pheno_pits_'+feature+'_'+database_parcel+'_Freesurfer_new/'
        working_dir = '/neurospin/brainomics/2016_HCP/new_analysis_threshold_'+feature_threshold+'/pits_analysis_'+feature+'_'+database_parcel+'_Freesurfer_new'
        output = '/neurospin/brainomics/2016_HCP/new_dictionaries_herit_threshold_'+feature_threshold+'/pits_'+feature+'_'+database_parcel+'_Freesurfer_new'
    
    if not os.path.exists(output):
        os.makedirs(output)

    if CC_ANALYSIS:
        output = output+"CC_"
        for side in sides:
            pheno = 'case_control_'+sds[side]
            dict_h2[pheno] = {}
            dict_pval[pheno] = {}
            filename = pheno_dir+'case_control/all_pits_side'+side+'.csv'
            df = pd.read_csv(filename)
            for col in df.columns:
                if col != 'IID' and col !='Age' and col != 'eTIV':
                    file_path = os.path.join(working_dir, pheno, col, 'polygenic.out')
                    m = re.search('Parcel_(.+?)end', col+'end')
                    if m:
                        num = m.group(1)
                    trait = str(int(num))
                    for line in open(file_path, 'r'):
                        if 'H2r is' in line and '(Significant)' in line:
                            print line[4:len(line)-15]
                            h2 = line[11:len(line)-30] 
                            p = line[26:len(line)-15]                
                            for k,l in enumerate(h2):
                                if not (l  in nb):
                                    break
                            h2 = float(h2[:k])
                            p = float(p)
                            print "We extracted h2: "+str(h2)+" pval: "+str(p)
                            dict_h2[pheno][trait] = h2
                            dict_pval[pheno][trait] = p
                            #dict_subj
    else:
        for side in sides:
            pheno = feature+"_"+sds[side]
            dict_h2[pheno] = {}
            dict_pval[pheno] = {}
            for filename in glob.glob(os.path.join(pheno_dir,'*.csv')):
                 if 'side'+side in filename:
                    m = re.search(pheno_dir+feature+'_pit(.+?)side'+side, filename)
                    if m:
                        num = m.group(1)

                    file_path = os.path.join(working_dir, pheno, "Parcel_"+str(num), 'polygenic.out')
                    trait = str(int(num))
                    for line in open(file_path, 'r'):
                        if 'H2r is' in line and '(Significant)' in line:
                            print line[4:len(line)-15]
                            h2 = line[11:len(line)-30] 
                            p = line[26:len(line)-15]                
                            for k,l in enumerate(h2):
                                if not (l  in nb):
                                    break
                            h2 = float(h2[:k])
                            p = float(p)
                            #if p<5e-2/10.0:
                            print "We extracted h2: "+str(h2)+" pval: "+str(p)
                            dict_h2[pheno][trait] = h2
                            dict_pval[pheno][trait] = p

    encoded = json.dumps(dict_h2)
    with open(output+'h2_dict.json', 'w') as f:
        json.dump(encoded, f)

    encoded = json.dumps(dict_pval)
    with open(output+'pval_dict.json', 'w') as f:
        json.dump(encoded, f)
