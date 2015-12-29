"""
Created  01 16 2015

@author vf140245
"""
import os
import optparse
import subprocess
import pandas as pd
import tempfile

if __name__ == "__main__":
    #linear = ('/neurospin/brainomics/2015_asym_sts/data/'
    #    'sts-sillons.STs_asym.assoc.linear')
    #linear = ('/neurospin/brainomics/2015_asym_sts/data/'
    #    '/STs_asym_rightonly_sts_gender_centre.STs_asym.assoc.linear')
    trait = ["STsR", "STsL", "STs_asym"]
    linear  = ('/neurospin/brainomics/2015_asym_sts/'
               'pheno/STs_depth_sts_gender_centre.'+trait[2]+'.assoc.linear')
    parser = optparse.OptionParser()
    parser.add_option('-l', '--linear',
                      help='path to linear plink file to parse',
                      default=linear, type="string")

    (options, args) = parser.parse_args()
    out = os.path.join(os.path.dirname(options.linear),
                       os.path.splitext(os.path.basename(options.linear))[0] +
                       '.pval')
    outsel = os.path.join(os.path.dirname(options.linear),
                       os.path.splitext(os.path.basename(options.linear))[0] +
                       '.sel')

    tmp = tempfile.mktemp()
    cmd = ["head -1 %s > %s" % (options.linear, tmp),
           ";",
           "grep ADD %s >> %s" % (options.linear, tmp)]
    print " ".join(cmd)
    p = subprocess.Popen(" ".join(cmd), shell=True)
    p.wait()
    cmd = ["awk '{print $1,$2,$3,$4,$5,$6,$7,$8,$9}' %s > %s " % (tmp, out)]
    print " ".join(cmd)
    #check_call
    p = subprocess.check_call(" ".join(cmd), shell=True)
    os.remove(tmp)
    pval = pd.io.parsers.read_csv(out, sep=' ')
    pvalsub = pval.loc[pval['P'] < 5e-5]
    print pvalsub

    pvalsub.to_csv(outsel,
                    sep='\t', index=False)
