# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:06:39 2013

@author: md238665

Some useful functions.

"""

import collections

import numpy
import sklearn.preprocessing
import pandas

import nibabel

import data_api

def n_indicator_columns(mapping):
    '''Determine the number of columns for indicator variables coding of a given categorical variable mapping:
         - if the values is None -> 1 column
         - else len(values) columns'''
    if mapping == None:
        return 1
    else:
        return len(mapping)

def indicator_variables(df, variables=None):
    '''Return a dataframe with indicator variables'''
    if not variables:
        variables = df.columns()
    mappings  = map(data_api.REGRESSOR_MAPPINGS.get, variables)
    n_cols = sum(map(n_indicator_columns, mappings))
    print "Expanding categorical variables yields %i columns" % n_cols

    series_dict = collections.OrderedDict()
    dummy_col_index = 0
    for variable, mapping in zip(variables, mappings):
        if mapping:
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
    if mapping == None:
        return 1
    else:
        return len(mapping) - 1

def dummy_coding(df, variables=None):
    '''Return a dummy coded dataframe'''
    if not variables:
        variables = df.columns()
    mappings  = map(data_api.REGRESSOR_MAPPINGS.get, variables)
    n_cols = sum(map(n_dummy_columns, mappings))
    print "Dummy coding categorical variables yields %i columns" % n_cols

    series_dict = collections.OrderedDict()
    dummy_col_index = 0
    for variable, mapping in zip(variables, mappings):
        if mapping:
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
    if not regressors:
        regressors = df.columns
    n_obs = df.shape[0]

    # Create design matrix
    if use_dummy_coding:
        design_mat = dummy_coding(df, regressors)
    else:
        design_mat = indicator_variables(df, regressors)

    # Add an intercept column & normalize (we loose dataframe here - well done)
    if scale:
        design_mat = sklearn.preprocessing.scale(design_mat)
    if intercept:
        design_mat = numpy.hstack((design_mat, numpy.ones((n_obs, 1))))

    return design_mat

def make_image_from_array(array, babel_mask):
    mask = babel_mask.get_data()
    binary_mask = mask!=0
    img = numpy.zeros(binary_mask.shape)
    img[binary_mask] = array
    outimg = nibabel.Nifti1Image(img, babel_mask.get_affine())
    return outimg