# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 13:47:36 2014

@author: christophe
"""

import matplotlib.pylab as plt


def f(df, column, global_pen='a', l1_pen='l1_ratio', tv_pen='tv_ratio'):
    fig_groups = df.groupby(global_pen)
    for global_pen_val, global_pen_group in fig_groups:
#        print ("gpg : ", global_pen_group)
#        print ("gpv : ", global_pen_val)
        plt.figure()
        plt.suptitle(str(global_pen_val))
        line_groups = global_pen_group.groupby(l1_pen)
        for l1_pen_val, l1_pen_group in line_groups:
#            print "l1 pen group", len(l1_pen_group)
            l1_pen_group.sort(tv_pen, inplace=True)
            l1_pen_group.plot(x=tv_pen, y=column, label=str(l1_pen_val))
            plt.xlabel('TV')
            plt.ylabel(column)
        plt.legend()
