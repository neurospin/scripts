# -*- coding: utf-8 -*-
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
#list_exponent = '⁰¹²³⁴⁵⁶⁷⁸⁹'
#sides = ['asym']
Bonf_correction = 120
verbose = False
if __name__ == '__main__': 
    """parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('-f', '--feature', type=str,
                        help="Features to measure")
    parser.add_argument('-s', '--symmetric', type=int,
                        help="Boolean, need to give 0 for False and another int for True " 
                        "Specify if symmetric template is used or not")
    parser.add_argument('-d', '--database', type=str,
                        help='Data base from which we take the cluster')
    options = parser.parse_args()
    ## INPUTS ###
    feature = options.feature    
    SYMMETRIC = bool(int(options.symmetric))
    database_parcel  = options.database"""

    feature = 'DPF'
    SYMMETRIC = True
    database_parcel = 'hcp'
    feature_threshold = 'DPF'

    if SYMMETRIC:
        pheno_dir = '/neurospin/brainomics/2016_HCP/new_pheno_threshold_'+feature_threshold+'/pheno_pits_sym_'+feature+'_'+database_parcel+'_Freesurfer_new/'
        working_dir = '/neurospin/brainomics/2016_HCP/new_analysis_threshold_'+feature_threshold+'/pits_analysis_sym_'+feature+'_'+database_parcel+'_Freesurfer_new'
    else:
        pheno_dir = '/neurospin/brainomics/2016_HCP/new_pheno_threshold_'+feature_threshold+'/pheno_pits_'+feature+'_'+database_parcel+'_Freesurfer_new/'
        working_dir = '/neurospin/brainomics/2016_HCP/new_analysis_threshold_'+feature_threshold+'/pits_analysis_'+feature+'_'+database_parcel+'_Freesurfer_new' 

    covariates = ['age*sex', 'age^2*sex', 'age^2','age', 'sex', 'etiv']
    elements = ['trait', 'nb_subj', 'h2', 'h2std', 'pval', 'h2cov']
    for side in sides:
        df = pd.DataFrame()
        dict_df = {}
        for elem in elements:
            dict_df[elem] = []
        for cov in covariates:
            dict_df[cov] = []
        pheno = feature+"_"+sds[side]
        count = 0
        for filename in glob.glob(os.path.join(pheno_dir,'*.csv')):
            cnt = 0
            if 'side'+side in filename:
                count += 1
                m = re.search(pheno_dir+feature+'_pit(.+?)side'+side, filename)
                if m:
                    num = m.group(1)

                file_path = os.path.join(working_dir, pheno, "Parcel_"+str(num), 'polygenic.out')
                trait = int(num)
                dict_pcov = {}
                for line in open(file_path, 'r'):
                    if 'H2r is' in line and '(Significant)' in line:                        
                        p = float(line[26:len(line)-15])
                        if p < 5e-2/Bonf_correction:
                            significant = True
                        else:
                            significant = False
                        break
                    else:
                        significant = False
                #significant = True
                if significant:
                    for line in open(file_path, 'r'):
                        if 'Trait' in line:                            
                            trait = trait #line[14:24]
                            dict_df['trait'].append(trait)                            
                            nb_subj = int(line[50:len(line)])
                            dict_df['nb_subj'].append(nb_subj)
                            if verbose:
                                print line
                                print 'Trait extracted: '+str(trait)
                                print 'Subjects extracted: '+ str(nb_subj)
                        elif 'H2r is' in line and '(Significant)' in line:
                            h2 = line[11:len(line)-30] 
                            if 'Not Significant' in line:
                                p = float(line[26:len(line)-19])
                            else:
                                p = float(line[26:len(line)-15])
                            for k,l in enumerate(h2):
                                if not (l  in nb):
                                    break
                            h2 = float(h2[:k])
                            p = float(p)
                            #if p<5e-2/10.0:
                            if verbose:
                                print line[4:len(line)-15]
                                print "We extracted h2: "+str(h2)+" pval: "+str(p)
                            dict_df['h2'].append(round(h2,2))
                            if p < 0.01:
                                p_temp = '%.1e'% p
                                """p_temp0= p_temp[:5]
                                for k,c in enumerate(p_temp[5:]):
                                    if k==0 and int(c)==0:
                                        pass
                                    else:
                                        p_temp0+=list_exponent[int(c)]"""
                                p_temp= p_temp.replace('e-0', '·10-')
                                p_temp = p_temp.replace('e-', '·10-')
                                dict_df['pval'].append(p_temp)
                            else:
                                dict_df['pval'].append(str(round(p,2)))
                            if h2 == 0:
                                dict_df['h2std'].append(0)
                        elif 'H2r Std.' in line:
                            h2std= float(line[25:len(line)])                        
                            if verbose:
                                print line
                                print 'Std extracted: '+str(h2std)
                            dict_df['h2std'].append(round(h2std,2))
                        elif 'Proportion of Variance' in line:
                            if verbose:
                                print line
                            cnt = 1
                        elif cnt == 1:
                            h2cov = float(line[6:15])
                            if verbose:
                                print line
                                print "we found h2cov: "+str(h2cov)
                            cnt = 0
                            dict_df['h2cov'].append(round(h2cov*100,1))
                        else:
                            #print "HERE"
                            for cov in covariates:
                                if cov in line and 'Significant' in line:

                                    if 'Not Significant' in line:
                                        p = float(line[47:len(line)-20])
                                    else:
                                        p = float(line[47:len(line)-16])
                                    if verbose:
                                        print line
                                        print 'pval found for '+cov+ ' '+str(p)
                                    if p < 0.01:
                                        p_temp = '%.1e'% p
                                        """p_temp0 = p_temp[:5]
                                        for k,c in enumerate(p_temp[5:]):
                                            if k==0 and int(c)==0:
                                                pass
                                            else:
                                                p_temp0+=list_exponent[int(c)]"""
                                        p_temp = p_temp.replace('e-0', '·10-')
                                        p_temp = p_temp.replace('e-', '·10-')
                                        dict_df[cov].append(p_temp)
                                    else:
                                        dict_df[cov].append(str(round(p,2)))
                                    break

                    # case when covariates explain 0% of variance
                    if len(dict_df['h2std']) > len(dict_df['h2cov']):
                        dict_df['h2cov'].append(0)
        for key in dict_df.keys():
            if verbose:
                print key
                print len(dict_df[key])
            df[key] = dict_df[key]

        labels = '/neurospin/brainomics/2016_HCP/LABELS/labelling_sym_template.csv'
        df_labels = pd.read_csv(labels)
        df_labels.index = df_labels['Parcel']
        array_labels = []
        for k, num in enumerate(df['trait']):
            array_labels.append(df_labels.loc[num]['Name'].replace('_',' ').replace('.',''))
        df['trait'] = array_labels
        columns_order = ['trait', 'h2', 'h2std', 'pval', 'age', 'age^2', 'sex', 'age*sex', 'age^2', 'etiv', 'h2cov', 'nb_subj']
        df = df[columns_order]
        df.columns = ['Trait', 'h²', 'std', 'p', 'Age(p)', 'Age²(p)', 'Sex(p)', 'Age*Sex(p)', 'Age²(p)', 'eTIV(p)', 'h²cov(%)', 'Subjects']
        OUTPUT = '/neurospin/brainomics/2016_HCP/LABELS/corrected_more_results_template_sym_'+side+'.csv'
        count = 0
        for trait in df['Trait']:
            if 'gyrus' in trait:
                count += 1
        print "Number of real clusters (not with 'gyrus') side "+side+' '+str(len(df['Trait'])-count)
        array_result = []
        for k in df.index:
             array_result.append(str(df.loc[k]['h²'])+'±'+str(df.loc[k]['std'])+' ('+str(df.loc[k]['p'])+')')
        df['h²±SE(p)'] = array_result
        df =df[['Trait', 'h²±SE(p)', 'Age(p)', 'Age²(p)', 'Sex(p)', 'Age*Sex(p)', 'Age²(p)', 'eTIV(p)', 'h²cov(%)', 'Subjects']]
        df.index = df['Trait']
        df= df.sort_index()
        df.to_csv(OUTPUT, sep=',', header=True, index=False)
        
