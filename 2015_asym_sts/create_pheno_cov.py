"""
Created  01 12 2015

@author vf140245
"""
import pandas as pd
import pheno
import optparse

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

    # get all qc data about sulci
    sulci_dataframe = qc_sulci_qc_subject(percent_tol=options.tol)

    # selection S.T.s   => phenotype
    feature = 'depthMax'
    sulcus = dict(zip(['S.T.s'], ['STs']))
    sulc_pheno = extract_from_sulci_df(sulci_dataframe, feature, sulcus)

    # now build the covar information  => covariate out
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
           'covar/sts_gender_centre.cov')
    pheno.to_PLINK_covar(covar[['FID', 'IID', 'Gender', 'City']], 
                         out, colnames=['Gender','City'])
    
    #save data for further processings
    out = ('/neurospin/brainomics/imagen_central/'
           'covar/covar_GenCitHan_GCTA.cov')
    pheno.to_GCTA_qcovar(covar, out)
    out = ('/neurospin/brainomics/imagen_central/'
           'covar/covar_GenCitHan_MEGHA.cov')
    pheno.to_MEGHA_covar(covar, out, colnames=['Gender', 'Handedness', 'City'])
    out = ('/neurospin/brainomics/2015_asym_sts/'
           'pheno/STs_tol%.2f.phe' % tol)

    print sulc_pheno.head()   
#    import numpy as np
#    sulc_pheno['STs_left_depthMax'] = np.log(sulc_pheno['STs_left_depthMax'])
#    sulc_pheno['STs_right_depthMax'] = np.log(sulc_pheno['STs_right_depthMax'])    
    print sulc_pheno.head()
    pheno.to_GCTA_pheno(sulc_pheno, out)
