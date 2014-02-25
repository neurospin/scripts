#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Set referentials
find . -name "*.nii"|while read f ; do /home/ed203246/git/scripts/tools/brainvisa/bin/set_transformations_referentials.py --input=$f ; done
"""

import os, sys, optparse
from soma import aims

def setref(input_filename, output_filename, referential):
    im = aims.read(input_filename)
    #print im.header()['referentials']
    #print im.header()['referentials'][0]
    #print im.header()['referentials'][1]
    for i in xrange(len(im.header()['referentials'])):
        #print i
        im.header()['referentials'][i] = referential
        im.header()['referentials'][i] = referential
    writer = aims.Writer()
    writer.write(im, output_filename)

if __name__ == "__main__":
    # Set default values to parameters
    referential = 'Talairach-MNI template-SPM'
    # parse command line options
    parser = optparse.OptionParser(description=__doc__)
    parser.add_option('--input',
        help='Input map volume', type=str)
    parser.add_option('--output',
        help='Output map volume, if missing ==output ', type=str)
    parser.add_option('--ref',
        help='Referential name (default="%s")' % 'Talairach-MNI template-SPM', type=str)
    options, args = parser.parse_args(sys.argv)
    #print __doc__
    if options.input is None:
        print "Error: Input is missing."
        parser.print_help()
        exit(-1)
    input_filename = options.input
    if options.output is None:
        output_filename = input_filename
    else:
        output_filename = options.output
    if options.ref is not None:
        referential = options.ref
    setref(input_filename, output_filename, referential)
    #print output_filename