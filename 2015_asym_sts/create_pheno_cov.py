"""
Created  01 12 2015

@author vf140245
"""
import pandas as pd
import pheno
import optparse
import re
import glob
import os
from qc_subjects import qc_sulci_qc_subject


def extract_from_sulci_df(sulci_dataframe, feature, sulcus):
    """
    """
    cnames = ['FID', 'IID']
    laterality = ['left', 'right']
    long_name = sulcus.keys()[0]
    short_name = sulcus.values()[0]
    cnames.extend(['mainmorpho_%s._%s.%s' % (long_name, i, feature)
                   for i in laterality])
    cols_laterality = dict(left=2, right=3)

    sulc_pheno = sulci_dataframe[cnames]
    left = sulci_dataframe[cnames[cols_laterality['left']]]
    right = sulci_dataframe[cnames[cols_laterality['right']]]
    asym_index = 2 * (left - right) / (left + right)
    asym_index = pd.DataFrame(dict(zip(['asym_%s_%s' % (short_name, feature)],
                                       [asym_index])),
                              index=sulc_pheno.index)
    sulc_pheno = pd.concat([sulc_pheno, asym_index], axis=1, join='inner')
    edit_col = [i.replace('mainmorpho_', '').replace(long_name, short_name).
                  replace('._', '_').replace('.', '_')
                for i in sulc_pheno.columns]
    sulc_pheno.columns = edit_col

    return sulc_pheno


# Run
#########################################
if __name__ == "__main__":
    """
    """
    #defaults
    tol = 0.02

    parser = optparse.OptionParser()
    parser.add_option('-t', '--tol',
                      help='Tolerance on the recog qual for sulci [%.2f]'%tol,
                      default=tol, type="float")
    (options, args) = parser.parse_args()

    #### CREATE COV ####
    # Actually the lines to build covariate files should rather be in a separate file create cov
    # Build the covar information  => covariate out
    # q quantitative
    # c categorical
    demo_csv = ('/neurospin/brainomics/2015_asym_sts/'
                'data/demographics.csv')
    
    q, c = pheno.readCovar(demo_csv)
    covar = c[['FID', 'IID', 'Gender', 'City',
               u'Handedness from QualityReport']]
    covar.columns = ['FID', 'IID', 'Gender', 'City', 'Handedness']
    covar = covar.dropna()
    out = ('/neurospin/brainomics/imagen_central/'
               'covar/gender_centre_handedness.cov')
    pheno.to_PLINK_covar(covar[['FID', 'IID', 'Gender', 'City', 'Handedness']], 
                         out, colnames=['Gender','City'])
    
    #save data for further processings
    out = ('/neurospin/brainomics/imagen_central/'
           'covar/covar_GenCitHan_GCTA.cov')
    pheno.to_GCTA_qcovar(covar, out)
    out = ('/neurospin/brainomics/imagen_central/'
           'covar/covar_GenCit_MEGHA.cov')
    pheno.to_MEGHA_covar(covar[['FID', 'IID', 'Gender', 'City']], out, colnames=['Gender', 'City'])



    ##### CREATE PHENO #####
    path = '/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/full_sulci'
    # get all qc data about sulci
    sulci_dataframe, sulcus_discarded = qc_sulci_qc_subject(percent_tol=options.tol)
    sulcus_discarded_names = []
    for j in range(len(sulcus_discarded)):
        m = re.search('mainmorpho_(.+?)._(.*).csv', sulcus_discarded[j]+'.csv')
        if m:
            sulcus = m.group(1)
            print "Discarded sulcus: " + str(sulcus)
            sulcus_discarded_names.append(sulcus)

    # selection S.T.s   => phenotype
    feature = 'depthMax'
    sulcus_list =  []
    for filename in glob.glob(os.path.join(path,'mainmorpho_*_right.csv')):
        print filename
        m = re.search('mainmorpho_(.+?)._right.csv', filename)
        if m:
            sulcus = m.group(1)
            print "Selected sulcus: " + str(sulcus)
        if sulcus not in sulcus_discarded_names:
            sulcus_list.append(dict(zip([sulcus], [sulcus.replace('.', '')])))
            print "Sulcus " + str(sulcus) + " has been done"
    #sulcus = dict(zip(['S.T.s'], ['STs']))
    for j in range(len(sulcus_list)):
        sulc_pheno = extract_from_sulci_df(sulci_dataframe, feature, sulcus_list[j])
        print sulc_pheno.head()   

        out = ('/neurospin/brainomics/2015_asym_sts/'
               'pheno/'+sulcus_list[j].values()[0]+'_tol%.2f.phe' % tol)
    
        #    import numpy as np
        #    sulc_pheno['STs_left_depthMax'] = np.log(sulc_pheno['STs_left_depthMax'])
        #    sulc_pheno['STs_right_depthMax'] = np.log(sulc_pheno['STs_right_depthMax'])    
        #    print sulc_pheno.head()
        #pheno.to_GCTA_pheno(sulc_pheno, out)
        
        out = ('/neurospin/brainomics/2015_asym_sts/'
               'PLINK_all_pheno'+str(tol)+'/'+sulcus_list[j].values()[0]+'_tol%.2f.phe' % tol)
        pheno.to_PLINK_pheno(sulc_pheno, out)
