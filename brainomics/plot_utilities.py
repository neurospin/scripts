# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 13:47:36 2014

@author: christophe
"""

import matplotlib.pylab as plt


def mapreduce_plot(df, column,
                   global_pen='a', l1_ratio='l1_ratio', tv_ratio='tv_ratio'):
    """Plot a metric of mapreduce as a function of the TV ratio.
    Data are grouped by global penalty (one global penalty per figure) and by
    value of the l1 ratio (one line per figure).
    If you have other parameters you must first group the results according to
    this parameter and then pass them to this function.

    Parameters
    ----------
    df: the dataframe containing the results.

    column: name of the column to plot.

    global_pen: name of the column containing global penalty or level for an
        Index.
        default: 'a'

    l1_ratio: name of the column containing the l1 ratio or level for an Index.
        default: 'l1_ratio'

    tv_ratio: name of the column containing the TV ratio or level for an Index.
        default: 'tv_ratio'

    Returns
    -------
    handles: dictionnary of figure handles.
        The key is the value of global_pen and the value is the figure handler.
    """
    # pandas.DataFrame.plot don't work properly if tv_ratio is an index
    import pandas.core.common as com
    tv_ratio_is_index = com.is_integer(tv_ratio)

    try:
        # Try to group by column name
        fig_groups = df.groupby(global_pen)
    except KeyError:
        try:
            # Try to group by index level
            fig_groups = df.groupby(level=global_pen)
        except:
            raise Exception("Canno't group by global_pen.")
    handles = {}
    for global_pen_val, global_pen_group in fig_groups:
        h = plt.figure()
        handles[global_pen_val] = h
        plt.suptitle(str(global_pen_val))
        try:
            # Try to group by column name
            line_groups = global_pen_group.groupby(l1_ratio)
        except KeyError:
            try:
                # Try to group by index level
                line_groups = global_pen_group.groupby(level=l1_ratio)
            except:
                raise Exception("Canno't group by l1_ratio.")
        for l1_ratio_val, l1_ratio_group in line_groups:
            if tv_ratio_is_index:
                name = l1_ratio_group.index.names[tv_ratio]
                l1_ratio_group.reset_index(tv_ratio, inplace=True)
                l1_ratio_group.plot(x=name, y=column,
                                    label=str(l1_ratio_val))
            else:
                l1_ratio_group.sort(tv_ratio, inplace=True)
                l1_ratio_group.plot(x=tv_ratio, y=column,
                                    label=str(l1_ratio_val))
            plt.xlabel('TV')
            plt.ylabel(column)
        plt.legend()
    return handles
