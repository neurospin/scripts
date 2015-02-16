#! /usr/bin/env python
##########################################################################
# Brainomics - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import patches


def genomic_plot(beta, genodata, selPathWay=None):
    """matplotlib plot of the estimated weights in genomic context

    Parameters:
    -----------
    beta: float array(should be same len as genodta.data.shape[1])
    genodata: the genotyping data and information
    selPW: int to sel the PW of interes
    """
    #int.
    color = '#e0eee0'

    groups_descr = genodata.get_meta_pws()
    groups_name = groups_descr.keys()
    groups = [list(groups_descr[n]) for n in groups_name]
    chrList = numpy.unique(genodata.get_meta()[1])
    selPW = numpy.zeros(genodata.data.shape[1])
    #si selection
    if selPathWay is not None:
        selPW[groups_descr[groups_name[selPathWay]]] = 1.
    lig = len(chrList) % 2 and (len(chrList) / 2 + 1) or len(chrList) / 2
    for i, chrom in enumerate(chrList):
        print " ", chrom
        # prepare subplot for chrom
        ax = plt.subplot(lig, 2, i)
        ax.set_yticklabels(())
        ax.set_xticklabels(())
        ax.text(.10, 0.10, chrom,
                horizontalalignment='center',
                transform=ax.transAxes)
        ax.title.set_visible(False)
        plt.ylim(-1, 1)
        # now get the data
        indx = numpy.where(numpy.asarray(genodata.get_meta()[1]) == chrom)[0]
        py = beta[indx]
        if selPW is not None:
            blackIndx = numpy.where(selPW[indx] == 0.)[0]
            redIndx = numpy.where(selPW[indx] != 0.)[0]
        px = range(len(py))
        plt.xlim(numpy.min(px) - 0.5, numpy.max(px) + 0.5)
        if len(blackIndx) != 0:
            plt.vlines(numpy.asarray(px)[blackIndx],
                       [0], numpy.asarray(py)[blackIndx], lw=2, colors='black')
        if len(redIndx) != 0:
            plt.vlines(numpy.asarray(px)[redIndx],
                       [0], numpy.asarray(py)[redIndx], lw=2, colors='red')
        # now plot the gene frontier
        allGenesLag = numpy.asarray(genodata.get_meta()[2])[indx]
        geneSet = set(allGenesLag)
        for g in geneSet:
            geneLag = numpy.where(allGenesLag == g)[0]
            if numpy.sum(selPW[indx][geneLag]) != 0.:  # gene bel. to sel'ed pw
                color = '#ffefd5'
            else:
                color = '#e0eee0'
            px = [numpy.min(geneLag) - 0.5, numpy.max(geneLag) + 0.5]
            py = [1, 1]
            plt.vlines(px, [-1], [1], lw=1, colors='white')
            p = patches.Rectangle((px[0], -1), px[1] - px[0], 2, color=color)
            ax.add_patch(p)
            ax.text(int(numpy.mean(px)), -0.5, g,
                horizontalalignment='center',
                fontsize=8, color='green')

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99,
                        bottom=0.01, wspace=0.01)
    plt.show()


def score_genodata(beta, genodata):
    """Details selection results by genes and pw

    Parameters:
    -----------
    """
    groups_descr = genodata.get_meta_pws()
    groups_name = groups_descr.keys()
    groups = [list(groups_descr[n]) for n in groups_name]
    ratioList = []
    ratioListGene = []
    for i in groups_descr:
        indx = groups_descr[i]
        sel = numpy.sum(beta[indx] != 0.)
        selIndx = numpy.where(beta[indx] != 0.)[0]
        print "==%s==\nselected %d out of %d SNPs (%f)" % \
                     (i, sel, len(indx), sel / (len(indx) * 1.))
        ratioList.extend([len(indx), sel / (len(indx) * 1.)])
        allGenesLag = numpy.asarray(genodata.get_meta()[2])[indx]
        text = []
        accGenSel = 0
        for g in set(allGenesLag):
            genSel = numpy.sum((numpy.array(allGenesLag)==g)[selIndx])
            genAll = numpy.sum((numpy.array(allGenesLag)==g))
            text.extend([g, ":%d/%d " % (genSel, genAll)])
            if genSel != 0:
                accGenSel +=1
        ratioListGene.extend([accGenSel/(len(allGenesLag) * 1.)])
        print " ".join(text)
        print "========================================"
#        print set(allGenesLag[selIndx])
#        print set(allGenesLag)
    bestpaw_asof_snp = numpy.argsort(ratioList)[-1]
    print "Selected pathway by SNP: %s" % groups_name[bestpaw_asof_snp]
    bestpaw_asof_gene = numpy.argsort(ratioListGene)[-1]
    print "Selected pathway by GENE: %s" % groups_name[bestpaw_asof_gene]
    print "BY SNP Worst to best pathway selected: ", numpy.argsort(ratioList)
    print "BY GENE Worst to best pathway selected: ", numpy.argsort(ratioListGene)
    return bestpaw_asof_snp, bestpaw_asof_gene

#load prepared data for the project (see exemple_pw.py)
fname = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/synaptic10.pickle'
fname = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/synapticAll.pickle'
f = open(fname)
genodata = pickle.load(f)
f.close()

# read x data
x = genodata.data
x_subj = ["%012d" % int(i) for i in genodata.fid]

# read y data
y = open('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/Hippocampus_L.csv').read().split('\n')[1:-1]
y_subj = [i.split('\t')[0] for i in y]
y = [float(i.split('\t')[2]) for i in y]

#intersect subject list
soi = list(set(x_subj).intersection(set(y_subj)))

# build daatset with X and Y
X = numpy.zeros((len(soi), x.shape[1]))
Y = numpy.zeros(len(soi))
for i, s in enumerate(soi):
    X[i, :] = x[x_subj.index(s), :]
    Y[i] = y[y_subj.index(s)]

# generate pseudo beta
beta = numpy.random.uniform(size=X.shape[1])
beta = beta * (1 * (beta >0.7))

best_pathway_asof_snp, best_pathway_asof_gene = score_genodata(beta, genodata)
genomic_plot(beta, genodata , best_pathway_asof_gene)


