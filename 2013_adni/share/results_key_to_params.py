# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 09:53:58 2014

@author: edouard.duchesnay@cea.fr
"""
import os, sys
import pandas as pd
import numpy as np
import argparse

params_columns = ["a", "l1", "l2", "tv", "k"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+', help='Input files')
    parser.add_argument('-o', '--output', help='Onput excl file')
    options = parser.parse_args()
    print options
    if not options.input:
        print 'Required arguments input'
        sys.exit(1)
    if not options.output:
        print 'Required arguments input'
        sys.exit(1)
    output_filename = options.output
    writer = pd.ExcelWriter(options.output)
    #df1.to_excel(writer, sheet_name='Sheet1')
    #df2.to_excel(writer, sheet_name='Sheet2')

    #output_filename = "/home/ed203246/Dropbox/results/adni/MCIc-MCInc_all.xls"
    #input_filenames = options.input
    #input_filename = "/home/ed203246/Dropbox/results/adni/MCIc-MCInc_cs.csv"
    for input_filename in options.input:
        print output_filename, input_filename
        #sys.exit(1)
        data = pd.read_csv(input_filename)
        params = pd.DataFrame([[float(p) for p in item.split("_")] for item in data.key],
                                columns = params_columns)
        data = pd.concat([data, params], axis=1)
        #data.to_csv("/tmp/toto.csv")
        data.to_csv(input_filename)
        name = os.path.splitext(os.path.basename(input_filename))[0]
        data.to_excel(writer, sheet_name=name, index=False)
    writer.close()