# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:42:32 2015

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import pandas
import pickle
import numpy



def init_data(SCRIPTDATA):
    # 1) Read imputed imagen data to compute maf from the Imagen DataSet
    #################################################################
    isnps = pickle.load(open(SCRIPTDATA+'/SNPheight_imputed.pickle'))
    maf = numpy.sum(isnps.data, axis=0) / (2. * isnps.data.shape[0])
    datas = {'snps': isnps.measure_ids,
             'Maf': maf}
    imagenImputed = pandas.DataFrame(datas, columns=['snps', 'Maf'],
                                     index=isnps.measure_ids)
    print ".. Read snp data ................................................"
    print "Sizes: ", imagenImputed.shape
    print "Columns: ", imagenImputed.columns.tolist()

    # 2) get the information from the plosOne paper.
    # comfront with IMAGEN imputed data
    #############################################
    plosList = SCRIPTDATA + '/SNPheight.csv'
    plosOne = pandas.DataFrame.from_csv(plosList, sep=';')
    plosOne['Beta'] = [float(i.replace('?', '-')) for i in plosOne['Beta']]
    plosOne['Freq'] = [float(i) for i in plosOne['Freq']]
    plosOne = pandas.merge(plosOne[['A1', 'A2', 'Freq', 'Beta']], imagenImputed,
                           left_index=True, right_index=True, how='inner')
    plosOne = plosOne.join(pandas.Series(numpy.abs(plosOne['Freq'] - plosOne['Maf']),
                                         name='Diff'))
    # reorder
    plosOne100 = plosOne.loc[isnps.measure_ids]
    print ".. Compare frequency from Zhang and Imagen ........................"
    print plosOne100[['Freq', 'Maf', 'Diff']].head()
    print " Statistics"
    print "Diff max: ", plosOne100['Diff'].max()
    print "Diff min: ", plosOne100['Diff'].min()
    print "Diff mean: ", plosOne100['Diff'].mean()
    print " Quantile 75% and 95%"
    q = plosOne100['Diff'].quantile([.75, .95])
    print q
    print " snps with difference in MAF in the 95% quantile"
    print plosOne100.loc[plosOne100['Diff'] > q[0.95]]
    plosOne95 = plosOne100.loc[plosOne100['Diff'] <= q[0.95]]
    
    # 3) Read data for covariate information
    ########################################
    fin = SCRIPTDATA + 'imagen_subcortCov_NP.csv'
    covar = pandas.read_csv(fin, sep=' ', dtype={0: str, 1: str})
    covar.index = covar['IID']
    covar = covar[['FID', 'IID', 'Age', 'ScanningCentre', 'Sex']]
    print " Read covar data ................................................"
    print covar.head()

    # 4) read height
    #############################################
    fname = SCRIPTDATA + '/height.phe'
    height = pandas.read_csv(fname, sep='\t', dtype={1: str, 0: str},
                             header=None)
    height.columns = ['FID', 'IID', 'height']
    height.index = height['IID']
    print " Read height data ................................................"
    print height.head()

    # 5) Mhippo
    #############################################
    fin = SCRIPTDATA + 'imagen_subcortCov_NP.csv'
    hippo = pandas.read_csv(fin, sep=' ', dtype={0: str, 1: str})
    hippo.index = covar['IID']
    hippo = hippo[['FID', 'IID', 'Mhippo', 'Lput', 'Lpal','Rput', 'Rpal']]
    print ".. Read hippocampus data .........................................."
    print hippo.head()

    # create the PgS :
    # snp_imagen x beta_plos
    # les rs ont été réordonnées see reorder above
    ###########################################################################
    beta = numpy.asarray(plosOne100['Beta']).reshape(173, -1)
    PgS = numpy.dot(isnps.data, beta).reshape(-1)
    studyPgS = pandas.DataFrame({'PgS': PgS, 'IID': isnps.subject_ids.reshape(-1)})
    studyPgS = pandas.merge(pandas.merge(covar, height, how='inner', on='IID'),
                            studyPgS, on='IID')
    studyPgS = pandas.merge(studyPgS, hippo, how='inner', on='IID')    
    studyPgSforR = studyPgS.copy()
    studyPgS = studyPgS[[u'FID_x', u'IID', u'Age', u'ScanningCentre',
                         u'Sex', u'height', u'PgS', u'Mhippo']]
    studyPgS.columns = [u'FID', u'IID', u'Age', u'ScanningCentre',
                        u'Sex', u'height', u'PgS', u'Mhippo']
    studyPgSforR[u'Sex'].replace({0: 'Male', 1: 'Female'}, inplace=True)
    studyPgSforR[u'ScanningCentre'].replace({1: 'Centre_1', 2: 'Centre_2',
                                             3: 'Centre_3', 4: 'Centre_4',
                                             5: 'Centre_5', 6: 'Centre_6',
                                             7: 'Centre_7', 8: 'Centre_8'},
                                            inplace=True)
    studyPgS.index = studyPgS['IID']

    return covar, height, hippo, studyPgS, isnps


def var_explained_pgs(covInfo, PgSInfo, col='height', covLabel=['Sex', 'Age']):
    covariate = numpy.matrix(pandas.get_dummies(PgSInfo['ScanningCentre'],
                                             prefix='Centre')[range(7)])
    covariate = numpy.hstack((covariate, numpy.asarray(PgSInfo[covLabel])))

    X = numpy.asarray(PgSInfo['PgS'])
    y = PgSInfo[col]
    X = X.reshape(-1, 1)
    subjects = PgSInfo['IID'].tolist()

    from sklearn.linear_model import LinearRegression
    Y = y - LinearRegression().fit(covariate, y).predict(covariate)
    lm = LinearRegression()
    lm.fit(X, Y)
    if col == 'height':
        print "..... Height related PgrS score explains ~ 6% of the height variability"
        print "      based on: ", X.shape[0], ' subjects'
        print "      covariate out is sex , age, scanning center"
    else:
        print "..... Part of variability explained by Height related polygenic score for %s"%col
        print "      based on: ", X.shape[0], ' subjects'
        print "      covariate out is sex , age, scanning center"
    print 'lm.score(X, Y)', lm.score(X, Y)

    return subjects, lm


def univariate(mask, snps, studyPgS, col='height'):
    # the SNP are X and reordered lines
    X = snps.data[mask, :]
    p = X.shape[1]
    permuter = snps.subject_ids[mask].tolist()
    y = studyPgS.loc[permuter][col]
    covariate = numpy.matrix(pandas.get_dummies(studyPgS.loc[permuter]['ScanningCentre'],
                                                  prefix='Centre')[range(7)])
    print "COVARIATE"
    print covariate
    covariate = numpy.hstack((covariate, numpy.asarray(studyPgS.loc[permuter][['Sex', 'Age']])))
    print "COVARIATE"
    print covariate
    
    from sklearn.linear_model import LinearRegression
    Y = y - LinearRegression().fit(covariate, y).predict(covariate)
    
    from scipy.stats import pearsonr
    pvals = []
    cors = []
    for i in range(X.shape[1]):
        cor, pval = pearsonr(X[:, i],  Y)
        cors.append(cor)
        pvals.append(pval)
    pvals = numpy.asarray(pvals)
    cors = numpy.asarray(cors)
    indices = numpy.where(pvals <= 0.05)
    print "\n"
    print "..... Univariate results"
    print '      numbers of significant p values *un*corrected', len(indices[0]), 'over ', p

    import p_value_correction as p_c
    p_corrected = p_c.fdr(pvals)
    w = numpy.where(p_corrected <= 0.05)[0]
    print '      numbers of significant corrected p values corrected', len(w), 'over ', p
    print '     ', snps.measure_ids[w], " pvalcor = ", p_corrected[w], " correlation = ", cors[w] 

    if col=='height':
        snps_mask = [snps.measure_ids.tolist().index(i) for i in snps.measure_ids[w]]
        subX = snps.data[mask, :][:, snps_mask]
        lm = LinearRegression()
        lm.fit(subX, Y)
        print "\n..... Score explained by the %d significant SNPS is ~ 1.5 percent of the height var"%len(w)
        print "      based on: ", subX.shape[0], ' subjects'
        print "      covariate out is sex , age, scanning center"
        print 'lm.score(X, Y)', lm.score(subX, Y)
    else:
        if len(w) > 0:
            print "\n..... Score explained by the %d significant SNPS of the  var" % (len(w), col)
            print "      based on: ", subX.shape[0], ' subjects'
            print "      covariate out is sex , age, scanning center"
            print 'lm.score(X, Y)', lm.score(subX, Y)
        else:
            print "\n..... Nothing in %s variability explained by this approach "% (col)

    return X, Y


def multivariate(X, Y, col='height'):
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
    if col == 'height':
        print "\n..... Score explained k=100 SNPS svr is ~ 1.8% of the height var"
        print "      SVR avec k best : eval CV 10 folds"
        print "      based on: ", X.shape[0], ' subjects'
        print "      covariate out is sex , age, scanning center"
        print cv_res
        print "\n      explained variance(mean): ", numpy.mean(cv_res)
    else:
        print "\n..... Score explained k=100 SNPS svr for the %s variability"%col
        print "      SVR avec k best : eval CV 10 folds"
        print "      based on: ", X.shape[0], ' subjects'
        print "      covariate out is sex , age, scanning center"
        print cv_res
        print "\n      explained variance(mean): ", numpy.mean(cv_res)
