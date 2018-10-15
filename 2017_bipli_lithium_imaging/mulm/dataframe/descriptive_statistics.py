# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 15:39:24 2014

@author:  edouard.duchesnay@gmail.com
@license: BSD-3-Clause

python ~/git/datamind/descriptive/descriptive_statistics.py -i /tmp/db.csv -o /tmp/db.xls -e "ID"
"""
import pandas as pd
import argparse, sys

def describe_df_basic(data):
    basic_desc = pd.DataFrame([[x, str(data[x].dtype), int(str(data[x].dtype)!="object"),
                          pd.notnull(data[x]).sum()] for x in data.columns],
                          columns=["variable", "type", "isnumeric", "count"])
    return basic_desc

def describe_df(data, exclude_from_cat_desc=[]):
    basic = describe_df_basic(data)
    desc_num = data.describe().T
    desc_num.insert(0, 'variable', desc_num.index)
    desc_num.index = range(len(desc_num))
    cat_vars = set(data.columns) - set(desc_num["variable"]) - set(exclude_from_cat_desc)
    #cat_vars = data.columns - desc_num["variable"] - exclude_from_cat_desc
    desc_cat = None
    for var in cat_vars:
        c = data[var].value_counts()
        d = pd.DataFrame(dict(variable=var, level=c.index, count=c))[["variable", "level", "count"]]
        d.index = range(len(d))
        desc_cat = d if desc_cat is None else desc_cat.append(d)
    return basic, desc_num, desc_cat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="Input csv file")
    parser.add_argument('-o', '--ouput', help="Input csv file")
    parser.add_argument('-e', '--exclude', help="variable to exclude (quoted, sep by space)")
    options = parser.parse_args()
    if not options.input or not options.ouput:
        parser.print_help()
        sys.exit(1)
    exclude = options.exclude
    if options.exclude is not None:
        exclude = options.exclude.split()
    else:
        exclude = []
    print(options.input, options.ouput, exclude)
    data = pd.read_csv(options.input)
    desc_num, desc_cat = describe_df(data, exclude=exclude)
    try:
        with pd.ExcelWriter(options.ouput) as writer:
            desc_num.to_excel(writer, sheet_name='Numercial')
            if desc_cat is not None:
                desc_cat.to_excel(writer, sheet_name='Categorial')
    except:
        out_num = options.ouput.replace(".xls", "_num.csv")
        desc_num.to_csv(out_num, index=False)
        if desc_cat is not None:
            out_cat = options.ouput.replace(".xls", "_cat.csv")
            desc_cat.to_csv(out_cat, index=False)
