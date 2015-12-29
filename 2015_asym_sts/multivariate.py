# -*- coding: utf-8 -*-
"""
Created on 26 october 2015

@author: yl247234
Copyrignt : CEA NeuroSpin - 2014
"""
import time
import os
import pandas
import json
import getpass
import numpy as np
import genibabel
print "genibabel version: " + str(genibabel.__version__)
from genibabel import imagen_genotype_measure

pheno_file = os.path.join('/neurospin/brainomics/2015_asym_sts/pheno',
                        'STs_asym.phe')

pheno_name = "STs_asym"

cov_file = os.path.join('/neurospin/brainomics/imagen_central/covar',
                        'sts_gender_centre.cov')

"""geno_file = os.path.join('/neurospin/brainomics/imagen_central/geno',
                         'qc_sub_qc_gen_all_snps_common_autosome')"""

def init_data():
    #set login, password, DB
    if 'KEYPASS' in os.environ:
        if os.path.isfile(os.environ['KEYPASS']):
            login = json.load(open(os.environ['KEYPASS']))['login']
    else:
        login = raw_input("\nImagen2 login: ")
    if 'KEYPASS' in os.environ:
        if os.path.isfile(os.environ['KEYPASS']):
            password = json.load(open(os.environ['KEYPASS']))['passwd']
    else:
        password = getpass.getpass("Imagen2 password: ")
    # 1) Read imagen data to compute maf from the Imagen DataSet
    t0 = time.time()
    snps = imagen_genotype_measure(login=login,
                                    password=password,
                                    all_measures=True)
    print "Elapsed time to retrieve all the snps from imagen2 to memory.. : " + str(time.time()-t0)
    maf = np.sum(snps.data, axis=0) / (2. * snps.data.shape[0])
    datas = {'snps': snps.measure_ids,
             'Maf': maf}
    snps_df = pandas.DataFrame(datas, columns=['snps', 'Maf'],
                                     index=snps.measure_ids)
    
    print ".. Read snp data ................................................"
    print "Sizes: ", snps_df.shape
    print "Columns: ", snps_df.columns.tolist()


    # 3) Read data for covariate information
    ########################################
    fin = cov_file
    covar = pandas.read_csv(fin, delim_whitespace = True, dtype={0: str, 1: str})
    covar.index = covar['IID']
    covar = covar[['FID', 'IID', 'City_DUBLIN',  'City_HAMBURG',  'City_LONDON',  'City_MANNHEIM',  'City_NOTTINGHAM', 'Gender_Female']]
    print " Read covar data ................................................"
    print covar.head()

    # 4) read Phenotype
    #############################################
    fname = pheno_file
    pheno_df = pandas.read_csv(fname, sep='\t', dtype={1: str, 0: str})
    pheno_df.index = pheno_df['IID']
    print " Read "+pheno_name+ " data ................................................"
    print pheno_df.head()

    return covar, pheno_df, snps




def multivariate(X, Y, col=pheno_name):
    from sklearn import cross_validation
    from sklearn import svm
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression

    anova_filter = SelectKBest(score_func=f_regression, k=100)
    #svr = svm.SVR(kernel="linear")
    svr = svm.SVR(kernel="rbf")
    anova_svr = Pipeline([('anova', anova_filter), ('svr', svr)])
    cv_res = cross_validation.cross_val_score(anova_svr, X, Y, cv=10)
    if col == pheno_name:
        print "\n..... Score explained k=100 SNPS svr is ~ 1.8% of the "+pheno_name+" var"
        print "      SVR avec k best : eval CV 10 folds"
        print "      based on: ", X.shape[0], ' subjects'
        print "      covariate out is sex , age, scanning center"
        print cv_res
        print "\n      explained variance(mean): ", np.mean(cv_res)
    else:
        print "\n..... Score explained k=100 SNPS svr for the %s variability"%col
        print "      SVR avec k best : eval CV 10 folds"
        print "      based on: ", X.shape[0], ' subjects'
        print "      covariate out is sex , age, scanning center"
        print cv_res
        print "\n      explained variance(mean): ", np.mean(cv_res)

def univariate(snps,X, Y, col=pheno_name):
    p = X.shape[1]
    from scipy.stats import pearsonr
    pvals = []
    cors = []
    for i in range(X.shape[1]):
        cor, pval = pearsonr(X[:, i],  Y)
        cors.append(cor)
        pvals.append(pval)
    pvals = np.asarray(pvals)
    cors = np.asarray(cors)
    indices = np.where(pvals <= 0.05)
    print "\n"
    print "..... Univariate results"
    print '      numbers of significant p values *un*corrected', len(indices[0]), 'over ', p

    import p_value_correction as p_c
    p_corrected = p_c.fdr(pvals)
    w = np.where(p_corrected <= 0.05)[0]
    print '      numbers of significant corrected p values corrected', len(w), 'over ', p
    print '     ', snps.measure_ids[w], " pvalcor = ", p_corrected[w], " correlation = ", cors[w] 

    """lm = LinearRegression()
    lm.fit(X, Y)
    print "\n..... Score explained by the %d significant SNPS is ~ 1.5 percent of the "+pheno_name+" var"%len(w)
    print "      based on: ", X.shape[0], ' subjects'
    print "      covariate out is sex, scanning center"
    print 'lm.score(X, Y)', lm.score(X, Y)
    else:
        if len(w) > 0:
            print "\n..... Score explained by the %d significant SNPS of the  var" % (len(w), col)
            print "      based on: ", subX.shape[0], ' subjects'
            print "      covariate out is sex , age, scanning center"
            print 'lm.score(X, Y)', lm.score(X, Y)
        else:
            print "\n..... Nothing in %s variability explained by this approach "% (col)
    """


if __name__ == "__main__":
    covar, pheno_df, snps = init_data()

    covLabel = ['City_DUBLIN',  'City_HAMBURG',  'City_LONDON',  'City_MANNHEIM',  'City_NOTTINGHAM', 'Gender_Female']
    subjects_cov = covar['IID'].tolist()
    subjects_X = snps.subject_ids.tolist()
    subjects_Y = pheno_df['IID'].tolist()
    subjects = [ subj for subj in subjects_cov if (subj in subjects_X and subj in subjects_Y)]
    mask_covar = [ subjects_cov.index(i) for i in subjects]
    covariate = np.asarray(covar[covLabel])
    covariate = covariate[mask_covar,:]
    mask_X = [snps.subject_ids.tolist().index(i) for i in subjects]
    X = snps.data[mask_X, :]
    mask_y = [ subjects_Y.index(i) for i in subjects]
    y = pheno_df[pheno_name][mask_y]

    from sklearn.linear_model import LinearRegression
    Y = y - LinearRegression().fit(covariate, y).predict(covariate)

    """lm = LinearRegression()

    print " Start the fitting: "
    t0 = time.time()
    lm.fit(X, Y)
    print "Elapsed time to fit Y.. : " + str(time.time()-t0) # 6 hours last time
    """

    multivariate(X, Y, col=pheno_name)
    univariate(snps,X, Y, col=pheno_name)
