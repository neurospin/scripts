# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:06:30 2015

@author: fh235918
"""

# System import
import numpy
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import patches


def genomic_plot(beta, genodata, pw_index=2):
    """matplotlib plot of the estimated weights in genomic context

    Parameters:
    -----------
    beta: float array(should be same len as genodata.data.shape[1])
    genodata: the genotyping data and information
    pw_index: int, index of pathway of interest
    """

    """ indexes_of_pw: dict that maps pathway name to indexes of snp measures
        associated to the pathway (indexes of columns in genodata)
    """
    indexes_of_pw = genodata.get_meta_pws()
    pw_names      = indexes_of_pw.keys()
    
    chrom_names   = list(set(genodata.get_meta()[1])) # kill redundancy
    chrom_to_int  = lambda x: 23 if x == "X" else 24 if x == "Y" else int(x)
    chrom_names   = sorted(chrom_names, key=chrom_to_int)
    nb_chroms     = len(chrom_names)
    
    # if selected pathway (pw)
    if pw_index is not None:
        pw_name     = pw_names[pw_index]
        _pw_indexes = indexes_of_pw[pw_name]
        pw_gnames   = set(numpy.asarray(genodata.get_meta()[2])[_pw_indexes])
    
    nb_rows = nb_chroms / 2 if nb_chroms % 2 == 0 else nb_chroms / 2 + 1
    
    y_max = 1.3*max(beta) # 10% bigger that max to keep space for gene name
    y_min = 0

    # set background of figure as white
    fig = plt.figure()
    fig.patch.set_facecolor("white")
    
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(nb_rows, 2)
    
    for i, chrom_name in enumerate(chrom_names):
        
        ax = plt.subplot(gs[int(i/2), i%2]) # row-ordered
        #ax = plt.subplot(gs[i % nb_rows, int(i/nb_rows)]) # column-ordered
        
        ax.set_yticklabels(())
        ax.set_xticklabels(())

        ax.text(1.0, 0.5, chrom_name, horizontalalignment="left",
                verticalalignment="center", fontsize=14, transform=ax.transAxes)
        
        #ax.title.set_visible(False)
        plt.ylim(y_min, y_max)
        ax.axhline(0, color="black", lw=2)

        # row label with chromosome names in genodata
        _chrom_names  = numpy.asarray(genodata.get_meta()[1])
        
        # chrom_indexes: columns in genodata for the current chromosome
        chrom_indexes = numpy.where(_chrom_names == chrom_name)[0]
        
        # gene names associated to the current chromosome in genodata
        chrom_gnames = numpy.asarray(genodata.get_meta()[2])[chrom_indexes]
        
        print chrom_name, ": ", len(chrom_indexes)

        ys       = (beta[chrom_indexes]) # absolute values
        nb_snps  = len(ys)
        nb_genes = len(set(chrom_gnames))
        
        total_space     = 1.0 # 100%
        total_intergene_space = 0.2 * total_space # 20% of space accorded to separate genes
        intergene_space = total_intergene_space / (nb_genes + 1.0)
        intersnp_space  = (total_space - total_intergene_space) / (nb_snps-nb_genes)

        #import pdb
        #pdb.set_trace()

        xs_snps        = []
        xs_genes       = []       # list of tuples (x start, x end, name)
        xs_intergenes  = []       # list of tuples (x start, x end)
        previous_gname = "blabla" # dumb initialization
        gene_start     = intergene_space # initialization: start of gene
        
        """ pw_indexes: indexes of snps of the pathway of interest in xs, ys 
            (for the current chromosome)
        """
        pw_indexes = []
        
        """ Building xs_snps, xs_intergenes, xs_genes:
            abscisses of snps, intergene regions and genes
        """
        for j in range(len(ys)):
            gname = chrom_gnames[j]
            if gname in pw_gnames:
                pw_indexes.append(j)
            if gname != previous_gname:
                previous_x = xs_snps[-1] if xs_snps else 0
                next_x     = previous_x + intergene_space
                xs_snps.append(next_x)
                xs_intergenes.append((previous_x, next_x))
                if previous_x != 0:
                    xs_genes.append((gene_start, previous_x, previous_gname))
                gene_start     = next_x # new gene just started
                previous_gname = gname  # for next iteration
            else:
                previous_x = xs_snps[-1]
                next_x     = previous_x + intersnp_space
                xs_snps.append(next_x)
        
        xs_genes.append((gene_start, next_x, gname))
        xs_intergenes.append((next_x, total_space))

        plt.xlim(0, total_space)
        
        assert len(xs_snps) == len(ys)
        
        plt.vlines(xs_snps, [0], ys, colors="black", linewidth=1)
        
        xs_selected = [xs_snps[_i] for _i in pw_indexes]
        ys_selected = [ys[_i]      for _i in pw_indexes]
        
        assert len(xs_selected) == len(ys_selected)
        
        plt.vlines(xs_selected, [0], ys_selected, linewidth=1, colors="red")
        
        """ intergene patches """
        for intergene_start, intergene_end in xs_intergenes:
            ax.add_patch(patches.Rectangle((intergene_start, y_min),
                                           intergene_end - intergene_start,
                                           y_max - y_min, fill=False,
                                           color="red", hatch="/"))        

        """ To avoid overlapping of gene names we use four different heights
            for gene names (y-axis positions). This generator returns 
            alternatively the four positions on the y-axis.
        """
        def alternate_gene_name_position(): # up and down to avoid overlapping
            while True:
                yield  y_max*.8
                yield  y_min - 0.15*y_max
                yield  y_max*1
                #yield -1.9
        
        
        """ gene patches and names """
        position = alternate_gene_name_position() # generator to get y position
        for gene_start, gene_end, gene_name in xs_genes:

            # color: yellow for the genes in the pathway of interest else green
            color = "yellow" if gene_name in pw_gnames else "#e0eee0"
            
            ax.add_patch(patches.Rectangle((gene_start, y_min),
                                           gene_end - gene_start,
                                           y_max - y_min, color=color))

            ax.text(numpy.mean((gene_start, gene_end)), position.next(),
                    gene_name, horizontalalignment="center", fontsize=8, color="blue")

#    plt.subplots_adjust(left=0.01, right=0.99, top=0.99,
#                        bottom=0.01, wspace=0.005)
    plt.subplots_adjust(left=0.01, right=0.98, top=0.98,
                        bottom=0.02, wspace=0.03, hspace=0.3) 
    plt.show()
