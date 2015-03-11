#! /usr/bin/env python
##########################################################################
# Brainomics - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import plinkio


from read_data import read_hippo_l1_gl_ovl

covariate, Lhippo, genotype, groups_descr = read_hippo_l1_gl_ovl(pname='Lhippo')

#######################
# get the usual matrices
########################
Y = Lhippo['Lhippo'].as_matrix()
tmp = list(covariate.columns)
#tmp.remove('FID')
#tmp.remove('IID')
#tmp.remove('AgeSq')
mycol = [u'Age', u'Sex', u'ICV',u'Centre_1', u'Centre_2', u'Centre_3', u'Centre_4', u'Centre_5', u'Centre_6', u'Centre_7']
Cov = covariate[mycol].as_matrix()
tmp = list(genotype.columns)
tmp.remove('FID')
tmp.remove('IID')
X = genotype[tmp].as_matrix()


# Vérfication par rapport a Plink
Lhippo.to_csv('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/testLhippo.phe',
              cols=['FID', 'IID', 'Lhippo'], header=True, sep=" ",index=False)
covariate.to_csv('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/test2.cov',
                 cols=['FID', 'IID'] + mycol, header=True, sep=" ", index=False)
with open('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/testList.snp', 'w') as fp:
    fp.write("\n".join(list(genodata.snpid))+'\n')



geno = plinkio.Genotype('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/qc_subjects_qc_genetics_all_snps_wave2')
geno.setOrderedSubsetIndiv(indx)
gt = geno.snpGenotypeByName('rs3755456')
from sklearn.linear_model import LinearRegression
design = np.hstack((gt, Cov))
lm = LinearRegression().fit(design,Y)
lm.coef_
#array([  1.21754013e+02,   1.17584563e-02,   4.52263468e+01,
#         1.57838287e-03,  -7.68448614e+00,   9.74415895e+01,
#         4.50441017e+01,   7.42802857e+01,   1.13834516e+01,
#        -5.00354645e+01,   1.12587936e+02])
######### les beta 7eme colonne sont "identiques"
#   2    rs3755456   75141327    C        ADD      798      121.8        3.969    7.882e-05
#   2    rs3755456   75141327    C        Age      798    0.01176       0.1112       0.9115
#   2    rs3755456   75141327    C        Sex      798      45.23         1.21       0.2268
#   2    rs3755456   75141327    C        ICV      798   0.001578        11.41    5.277e-28
#   2    rs3755456   75141327    C   Centre_1      798     -7.684      -0.1109       0.9117
#   2    rs3755456   75141327    C   Centre_2      798      97.44        2.007      0.04511
#   2    rs3755456   75141327    C   Centre_3      798      45.04       0.6313        0.528
#   2    rs3755456   75141327    C   Centre_4      798      74.28        1.403       0.1609
#   2    rs3755456   75141327    C   Centre_5      798      11.38        0.226       0.8212
#   2    rs3755456   75141327    C   Centre_6      798     -50.04      -0.9059       0.3653
#   2    rs3755456   75141327    C   Centre_7      798      112.6        2.093      0.03667
