# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 13:47:36 2014

@author: christophe
"""

import matplotlib.pylab as plt


def plot_lines(df, x_col, y_col, colorby_col=None,
               splitby_col=None, color_map=None):
    """Plot a metric of mapreduce as a function of the TV ratio.
    Data are grouped by global penalty (one global penalty per figure) and by
    value of the l1 ratio (one line per figure).
    If you have other parameters you must first group the results according to
    this parameter and then pass them to this function.

    Parameters
    ----------
    df: the dataframe containing the results.

    x_col: column name of the x-axis.

    y_col: column name of the y-axis.

    colorby_col: column name that define colored lines. One line per level.

    splitby_col: column name that define how to split axis/figures.
    One axis/figure per level.

    Returns
    -------
    handles: dictionnary of figure handles.
        The key is the value of splitby_col and the value is the figure handler.
    """
    # pandas.DataFrame.plot don't work properly if x_col is an index
    import pandas.core.common as com
    x_col_is_index = com.is_integer(x_col)

    try:
        # Try to group by column name
        fig_groups = df.groupby(splitby_col)
    except KeyError:
        try:
            # Try to group by index level
            fig_groups = df.groupby(level=splitby_col)
        except:
            raise Exception("Cannot group by splitby_col.")
    handles = {}
    for splitby_col_val, splitby_col_group in fig_groups:
        h = plt.figure()
        handles[splitby_col_val] = h
        plt.suptitle(str(splitby_col_val))
        try:
            # Try to group by column name
            color_groups = splitby_col_group.groupby(colorby_col)
        except KeyError:
            try:
                # Try to group by index level
                color_groups = splitby_col_group.groupby(level=colorby_col)
            except:
                raise Exception("Cannot group by colorby_col.")
        for colorby_col_val, colorby_col_group in color_groups:
            if x_col_is_index:
                name = colorby_col_group.index.names[x_col]
                colorby_col_group.reset_index(x_col, inplace=True)
                colorby_col_group.plot(x=name, y=y_col,
                                       label=str(colorby_col_val))
            else:
                colorby_col_group.sort(x_col, inplace=True)
                if color_map is None:
                    colorby_col_group.plot(x=x_col, y=y_col,
                                           label=str(colorby_col_val))
                else:
                    colorby_col_group.plot(x=x_col, y=y_col,
                                           label=str(colorby_col_val),
                                           color=color_map[colorby_col_val])
            plt.xlabel(x_col)
            plt.ylabel(y_col)
        plt.legend()
    return handles
