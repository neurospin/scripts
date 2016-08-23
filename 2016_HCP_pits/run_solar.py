"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""

import os, glob, re, argparse
import pandas as pd
import time

CC_ANALYSIS = False # Soon deprecated usage
sides = ['R', 'L']
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
        working_dir = 'new_analysis_threshold_'+feature_threshold+'/pits_analysis_sym_'+feature+'_'+database_parcel+'_Freesurfer_new'
    else:
        pheno_dir = '/neurospin/brainomics/2016_HCP/new_pheno_threshold_'+feature_threshold+'/pheno_pits_'+feature+'_'+database_parcel+'_Freesurfer_new/'
        working_dir = 'new_analysis_threshold_'+feature_threshold+'/pits_analysis_'+feature+'_'+database_parcel+'_Freesurfer_new'

    os.system("solar makeped "+ working_dir)

    if CC_ANALYSIS:
        for side in sides:
            for i in range(5):
                print '\n'
            print "====================================Side " +sds[side]+"========================================================"
            for i in range(5):
                print '\n'
            output_dir= 'case_control_'+sds[side]
            filename = pheno_dir+"case_control/all_pits_side"+side+".csv"
            df = pd.read_csv(filename)
            for col in df.columns:
                if col != 'IID':
                    os.system("solar pheno_analysis "+working_dir+" "+output_dir+" "+col+" "+filename)
                    #time.sleep(1)
    else:
        for side in sides:
            output_dir = feature+"_"+sds[side]
            for i in range(5):
                print '\n'
            print "====================================Side " +sds[side]+"========================================================"
            for i in range(5):
                print '\n'
            for filename in glob.glob(os.path.join(pheno_dir,'*.csv')):
                if 'side'+side in filename:
                    m = re.search(pheno_dir+feature+'_pit(.+?)side'+side, filename)
                    if m:
                        num = m.group(1)

                    os.system("solar pheno_analysis "+working_dir+" "+output_dir+" Parcel_"+str(num)+" "+filename)
                    #time.sleep(1)


