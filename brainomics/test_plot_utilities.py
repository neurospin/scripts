# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 19:01:12 2014

@author: md238665

Test of plot_utilities.

"""

import numpy as np
import pandas as pd

from brainomics import plot_utilities

import matplotlib.pylab as plt

# Create a df
global_pens = [1, 10, 100]
tv_ratios = [0, 0.1, 0.5, 1.0]
l1_ratios = [0, 0.1, 0.5, 1.0]
index = np.array([[global_pen, tv_ratio, l1_ratio]
          for global_pen in global_pens
          for tv_ratio in tv_ratios
          for l1_ratio in l1_ratios])
df = pd.DataFrame(index, columns=['global_pen', 'tv_ratio', 'l1_ratio'])


def metric_fun(r):
    return (r['tv_ratio'] + r['l1_ratio']) * r['global_pen']

df['metric'] = df.apply(metric_fun, axis=1)

# Test
handles = plot_utilities.plot_lines(df,
                                    x_col='tv_ratio',
                                    y_col='metric',
                                    splitby_col='global_pen',
                                    colorby_col='l1_ratio',
                                    use_suptitle=False,
                                    use_subplots=True)
plt.show()

# Test with index
df.set_index(['global_pen', 'tv_ratio', 'l1_ratio'], inplace=True)
handles = plot_utilities.plot_lines(df,
                                    x_col=1,
                                    y_col='metric',
                                    splitby_col=0,
                                    colorby_col=2,
                                    use_suptitle=False,
                                    use_subplots=True)

plt.show()
