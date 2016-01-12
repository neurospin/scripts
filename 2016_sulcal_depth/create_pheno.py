"""
@author yl247234 
"""
import pandas as pd
import pheno
import optparse
import re, glob, os
from qc_subjects import qc_sulci_qc_subject

## INPUTS ##
TOLERANCE_THRESHOLD = 0.05
DIRECTORY_SULCI = '/volatile/yann/sulci_data/all_sulci/BL'
## OUTPUT ##
DIRECTORY_PHENOTYPE = '/volatile/yann/2016_sulcal_depth/pheno/'

def extract_from_sulci_df(sulci_dataframe, feature, sulcus):
    """
    """
    cnames = ['FID', 'IID']
    laterality = ['left', 'right']
    long_name = sulcus.keys()[0]
    short_name = sulcus.values()[0]
    cnames.extend(['morpho_%s._%s.%s' % (long_name, i, feature)
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
    edit_col = [i.replace('morpho_', '').replace(long_name, short_name).
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
    tol = TOLERANCE_THRESHOLD

    parser = optparse.OptionParser()
    parser.add_option('-t', '--tol',
                      help='Tolerance on the recog qual for sulci [%.2f]'%tol,
                      default=tol, type="float")
    (options, args) = parser.parse_args()

    path = DIRECTORY_SULCI
    # get all qc data about sulci
    sulci_dataframe, sulcus_discarded = qc_sulci_qc_subject(percent_tol=options.tol)
    sulcus_discarded_names = []
    for j in range(len(sulcus_discarded)):
        m = re.search('morpho_(.+?)._(.*).csv', sulcus_discarded[j]+'.csv')
        if m:
            sulcus = m.group(1)
            print "Discarded sulcus: " + str(sulcus)
            sulcus_discarded_names.append(sulcus)

    # selection S.T.s   => phenotype
    feature = 'depthMax'
    sulcus_list =  []
    for filename in glob.glob(os.path.join(path,'morpho_*_right.csv')):
        #print filename
        m = re.search('morpho_(.+?)._right.csv', filename)
        if m and os.path.isfile(os.path.join(path,'morpho_'+ m.group(1)+'._left.csv')):
            sulcus = m.group(1)
            #print "Selected sulcus: " + str(sulcus)
        if sulcus not in sulcus_discarded_names:
            sulcus_list.append(dict(zip([sulcus], [sulcus.replace('.', '')])))
            #print "Sulcus " + str(sulcus) + " has been done"
    for j in range(len(sulcus_list)):
        sulc_pheno = extract_from_sulci_df(sulci_dataframe, feature, sulcus_list[j])  

        out = (DIRECTORY_PHENOTYPE
               'PLINK_all_pheno'+str(tol)+'/'+sulcus_list[j].values()[0]+'_tol%.2f.phe' % tol)
        
        pheno.to_PLINK_pheno(sulc_pheno, out)
