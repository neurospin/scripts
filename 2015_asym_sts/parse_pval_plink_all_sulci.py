"""
Created  01 16 2015

@author vf140245
"""
import os, glob, re
import optparse
import subprocess
import pandas as pd
import tempfile

if __name__ == "__main__":
   
    path = '/neurospin/brainomics/2015_asym_sts/all_sulci_pvals/'
    features = ["depthMax"]
    pheno = ["right", "left", "asym"]
    sulcus_names = []
    for filename in glob.glob(os.path.join(path,'*_left_depthMax.assoc.linear')):
        m = re.search(path+'(.+?)_tol0.02(.*)_left_depthMax.assoc.linear', filename)
        if m:
            sulcus = m.group(1)
            #print "Sulcus: " + str(sulcus)
            sulcus_names.append(sulcus)
    for j in range(len(sulcus_names)):
        for i in range(len(pheno)):
            if i == 2:
                if sulcus_names[j] == 'SRinf' or sulcus_names[j] == 'SPasup' or  sulcus_names[j] =='SpC' or sulcus_names[j] =='SOp' or sulcus_names[j] =='SForbitaire' or sulcus_names[j]=='FIPrint1' or sulcus_names[j] == 'FCLrsc' or sulcus_names[j] == 'FCLrretroCtr':
                    break
                else:
                    linear  = path+sulcus_names[j]+'_tol0.02_gender_centre.'+pheno[i]+'_'+sulcus_names[j]+'_depthMax.assoc.linear'
            else:
                linear  = path+sulcus_names[j]+'_tol0.02_gender_centre.'+sulcus_names[j]+'_'+pheno[i]+'_depthMax.assoc.linear'
                
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
                                  '.sel7')

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
            pvalsub = pval.loc[pval['P'] < 5e-7]
            print pvalsub

            pvalsub.to_csv(outsel,
                           sep='\t', index=False)
