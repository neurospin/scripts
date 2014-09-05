# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:06:39 2013

@author: md238665

Some useful functions for pandas dataframe.

"""

import collections as coll

import numpy
import pandas as pd


def numerical_coding(df, mappings):
    """Return a dataframe where categorical variables are mapped to their
    numerical values.
    Parameters
    ----------
    df: the pandas DataFrame
    mappings: the correspondence (dict of OrderedDict)

    Returns
    -------
    new_df: the modififed DataFrame

    >>> s = pd.Series(numpy.random.randn(5))
    >>> s2 = pd.Series(['a', 'b', 'c', 'b', 'a'])
    >>> s3 = pd.Series(['apple', 'banana', 'orange', 'banana', 'apple'])
    >>> df = pd.DataFrame.from_dict({'cont': s, 'cat1': s2, 'cat2': s3})
    >>> mappings = {'cat1': coll.OrderedDict([('a', 1), ('b', 2), ('c', 3)])}
    >>> new_df = numerical_coding(df, mappings)
    >>> new_df['cat1']
    0    1
    1    2
    2    3
    3    2
    4    1
    Name: cat1, dtype: int64"""
    series_dict = coll.OrderedDict()
    for variable in df.columns:
        if variable in mappings:
            series_dict[variable] = df[variable].map(mappings[variable])
        else:
            series_dict[variable] = df[variable]
    return pd.DataFrame(series_dict)


def n_indicator_columns(mapping):
    """Determine the number of columns for indicator variables coding of a
    given categorical variable mapping:
      - if the values is None -> 1 column
      - else len(values) columns

    Parameters
    ----------
    mapping: map for a categorical variable

    Returns
    -------
    n: number of indicator columns for this variable

    >>> mapping = coll.OrderedDict([('a', 1), ('b', 2), ('c', 3)])
    >>> n_indicator_columns(mapping)
    3"""
    if mapping is None:
        return 1
    else:
        return len(mapping)


def indicator_variables(df, mappings):
    """Return a dataframe where categorical variables are replaced by indicator
    variables
    Parameters
    ----------
    df: the pandas DataFrame
    mapping = coll.OrderedDict([('a', 1), ('b', 2), ('c', 3)])

    Returns
    -------
    new_df: the modififed DataFrame

    >>> s = pd.Series(numpy.random.randn(5))
    >>> s2 = pd.Series(['a', 'b', 'c', 'b', 'a'])
    >>> s3 = pd.Series(['apple', 'banana', 'orange', 'banana', 'apple'])
    >>> df = pd.DataFrame.from_dict({'cont': s, 'cat1': s2, 'cat2': s3})
    >>> mappings = {'cat1': coll.OrderedDict([('a', 1), ('b', 2), ('c', 3)]), \
                    'cat2': coll.OrderedDict([('apple', 0), \
                                              ('banana', 2)])}
    >>> new_df = indicator_variables(df, mappings)
    >>> new_df['cat1#a']
    0    1
    1    0
    2    0
    3    0
    4    1
    Name: cat1#a, dtype: int64
    >>> new_df.shape[1] == sum(map(n_indicator_columns, mappings.values())) + \
                           df.shape[1] - len(mappings)
    True"""
    #n_cols = sum(map(n_indicator_columns, mappings))
    #print "Expanding categorical variables yields %i columns" % n_cols

    series_dict = coll.OrderedDict()
    dummy_col_index = 0
    for variable in df.columns:
        if variable in mappings:
            mapping = mappings[variable]
            categorical_values = mapping.keys()
            numerical_values = mapping.values()
            zipped = zip(numerical_values, categorical_values)
            for level_index, (level, category) in enumerate(zipped):
                dummy_var_name = '{cat}#{value}'.format(cat=variable,
                                                        value=category)
                #print "Putting %s at col %i" % (dummy_var_name,
                #                                dummy_col_index)
                column_serie = (df[variable] == category).astype(numpy.int64)
                series_dict[dummy_var_name] = column_serie
                dummy_col_index += 1
        else:
            #print "Putting ordinal variable %s at col %i" % (variable,
            #                                                 dummy_col_index)
            series_dict[variable] = df[variable]
            dummy_col_index += 1
    return pd.DataFrame(series_dict)


def n_dummy_columns(mapping):
    """Determine the number of columns for dummy coding of a given categorical
    variable mapping:
      - if the values is None -> 1 column
      - else len(values)-1 columns

    Parameters
    ----------
    mapping: map for a categorical variable

    Returns
    -------
    n: number of dummy columns for this variable

    >>> mapping = mapping = coll.OrderedDict([('a', 1), ('b', 2), ('c', 3)])
    >>> n_dummy_columns(mapping)
    2"""
    if mapping is None:
        return 1
    else:
        return len(mapping) - 1


def dummy_coding(df, mappings):
    """Return a dataframe where categorical variables are replaced by indicator
    variables
    Parameters
    ----------
    df: the pandas DataFrame
    mapping = coll.OrderedDict([('a', 1), ('b', 2), ('c', 3)])

    Returns
    -------
    new_df: the modififed DataFrame

    >>> s = pd.Series(numpy.random.randn(5))
    >>> s2 = pd.Series(['a', 'b', 'c', 'b', 'a'])
    >>> s3 = pd.Series(['apple', 'banana', 'orange', 'banana', 'apple'])
    >>> df = pd.DataFrame.from_dict({'cont': s, 'cat1': s2, 'cat2': s3})
    >>> mappings = {'cat1': coll.OrderedDict([('a', 1), ('b', 2), ('c', 3)]), \
                    'cat2': coll.OrderedDict([('apple', 0), \
                                              ('banana', 2)])}
    >>> new_df = dummy_coding(df, mappings)
    >>> new_df.shape[1] == sum(map(n_dummy_columns, mappings.values())) + \
                           df.shape[1] - len(mappings)
    True"""
    #n_cols = sum(map(n_dummy_columns, mappings))
    #print "Dummy coding categorical variables yields %i columns" % n_cols

    series_dict = coll.OrderedDict()
    dummy_col_index = 0
    for variable in df.columns:
        if variable in mappings:
            mapping = mappings[variable]
            categorical_values = mapping.keys()
            categorical_ref_value = categorical_values.pop(0)
            numerical_values = mapping.values()
            numerical_ref_value = numerical_values.pop(0)
            #print "Reference value for %s is %s (%d)" % (variable,
            #                                             categorical_ref_value,
            #                                             numerical_ref_value)
            zipped = zip(numerical_values, categorical_values)
            for level_index, (level, category) in enumerate(zipped):
                dummy_var_name = '{cat}#{value}'.format(cat=variable,
                                                        value=category)
                #print "Putting %s at col %i" % (dummy_var_name,
                #                                dummy_col_index)
                column_serie = (df[variable] == category).astype(numpy.int64)
                series_dict[dummy_var_name] = column_serie
                dummy_col_index += 1
        else:
            #print "Putting ordinal variable %s at col %i" % (variable,
            #                                                 dummy_col_index)
            series_dict[variable] = df[variable]
            dummy_col_index += 1
    return pd.DataFrame(series_dict)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
