# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:29:20 2014

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""

#grep chr22 /neurospin/bioinformactics_resource/data/deCODE/male.rgmap > extract.txt
a = open("/tmp/vf/extract.txt").read().split('\n')[:-1]

a = [i.split('\t')for i in a]
a = [[int(i[1]), float(i[3])] for i in a]
import matplotlib.pylab as plt
import numpy as np
a = np.asarray(a)

#grep ^22 /neurospin/bioinformactics_resource/data/genomic_plateform_resource/wave1/*bim > 610.txt
s = open("/tmp/vf/610.txt").read().split('\n')[:-1]
s = [int(i.split('\t')[3]) for i in s]

aM = np.max(a[:,0])
am = np.min(a[:,0])


plt.xlim(am,aM)
#plt.plot(a[:,0], a[:,1])
plt.vlines(a[:,0], [0], a[:,1], 'r')
plt.vlines(s, [0], np.ones(len(s)))
plt.show()

