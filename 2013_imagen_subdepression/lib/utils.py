# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:06:39 2013

@author: md238665

Some useful functions.

"""

import collections

import numpy
import pandas

import nibabel

import data_api

def numerical_coding(df, variables=None):
    '''Return a dataframe where categorical variables are mapped to their numerical values'''
    if variables is None:
        variables = df.columns()
    mappings  = map(data_api.REGRESSOR_MAPPINGS.get, variables)
    series_dict = collections.OrderedDict()
    for variable, mapping in zip(variables, mappings):
        if mapping is not None:
            series_dict[variable] = df[variable].map(data_api.GroupMap)
        else:
            series_dict[variable] = df[variable]
    return pandas.DataFrame(series_dict)
            

def n_indicator_columns(mapping):
    '''Determine the number of columns for indicator variables coding of a given categorical variable mapping:
         - if the values is None -> 1 column
         - else len(values) columns'''
    if mapping is None:
        return 1
    else:
        return len(mapping)

def indicator_variables(df, variables=None):
    '''Return a dataframe with indicator variables'''
    if variables is None:
        variables = df.columns()
    mappings  = map(data_api.REGRESSOR_MAPPINGS.get, variables)
    n_cols = sum(map(n_indicator_columns, mappings))
    print "Expanding categorical variables yields %i columns" % n_cols

    series_dict = collections.OrderedDict()
    dummy_col_index = 0
    for variable, mapping in zip(variables, mappings):
        if mapping is not None:
            categorical_values = mapping.keys()
            numerical_values = mapping.values()
            for level_index, (level, category) in enumerate(zip(numerical_values, categorical_values)):
                dummy_var_name = '{cat}#{value}'.format(cat=variable, value=category)
                print "Putting %s at col %i" % (dummy_var_name, dummy_col_index)
                column_serie = (df[variable] == category).astype(numpy.float64)
                series_dict[dummy_var_name] = column_serie
                dummy_col_index += 1
        else:
            print "Putting ordinal variable %s at col %i" % (variable, dummy_col_index)
            series_dict[variable] = df[variable].astype(numpy.float64)
            dummy_col_index += 1
    return pandas.DataFrame(series_dict)

def n_dummy_columns(mapping):
    '''Determine the number of columns for dummy coding of a given categorical variable mapping:
         - if the values is None -> 1 column
         - else len(values)-1 columns'''
    if mapping is None:
        return 1
    else:
        return len(mapping) - 1

def dummy_coding(df, variables=None):
    '''Return a dummy coded dataframe'''
    if variables is None:
        variables = df.columns()
    mappings  = map(data_api.REGRESSOR_MAPPINGS.get, variables)
    n_cols = sum(map(n_dummy_columns, mappings))
    print "Dummy coding categorical variables yields %i columns" % n_cols

    series_dict = collections.OrderedDict()
    dummy_col_index = 0
    for variable, mapping in zip(variables, mappings):
        if mapping is not None:
            categorical_values = mapping.keys()
            categorical_ref_value = categorical_values.pop(0)
            numerical_values = mapping.values()
            numerical_ref_value = numerical_values.pop(0)
            print "Reference value for %s is %s (%d)" % (variable, categorical_ref_value, numerical_ref_value)
            for level_index, (level, category) in enumerate(zip(numerical_values, categorical_values)):
                dummy_var_name = '{cat}#{value}'.format(cat=variable, value=category)
                print "Putting %s at col %i" % (dummy_var_name, dummy_col_index)
                column_serie = (df[variable] == category).astype(numpy.float64)
                series_dict[dummy_var_name] = column_serie
                dummy_col_index += 1
        else:
            print "Putting ordinal variable %s at col %i" % (variable, dummy_col_index)
            series_dict[variable] = df[variable].astype(numpy.float64)
            dummy_col_index += 1
    return pandas.DataFrame(series_dict)

def make_design_matrix(df, regressors=None, intercept=True, scale=False, use_dummy_coding=True):
    """Return a design matrix from the data frame.
    >>> df = pandas.DataFrame({'Gender': ['Male', 'Female', 'Male']})
    >>> design_mat = make_design_matrix(df, ['Gender'])
    Dummy coding categorical variables yields 1 columns
    Reference value for Gender is Male (0)
    Putting Gender#Female at col 0
    """
    if regressors is None:
        regressors = df.columns
    n_obs = df.shape[0]

    # Create design matrix
    if use_dummy_coding:
        design_mat = dummy_coding(df, regressors)
    else:
        design_mat = indicator_variables(df, regressors)

    # Add an intercept column & normalize (we loose dataframe here - well done)
    if scale:
        design_mat = (design_mat - design_mat.mean()) / (design_mat.max() - design_mat.min())
    if intercept:
        design_mat['Intercept'] = numpy.ones((n_obs, 1))

    return design_mat

def make_image_from_array(array, babel_mask):
    mask = babel_mask.get_data()
    binary_mask = mask!=0
    img = numpy.zeros(binary_mask.shape)
    img[binary_mask] = array
    outimg = nibabel.Nifti1Image(img, babel_mask.get_affine())
    return outimg

if __name__ == '__main__':
    import doctest
    doctest.testmod()