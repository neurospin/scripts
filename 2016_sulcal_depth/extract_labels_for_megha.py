

import numpy as np
import glob, os, re
import pickle


labels = []
path = '/neurospin/imagen/BL/processed/freesurfer/'
path_saved = os.getcwd()+'/megha/'
filename = 'labels_filtered.txt'

# Create the file fist else not all information are written
thefile = open(path_saved+ filename, 'wb')
thefile.close


for filename in glob.glob(os.path.join(path,'*')):
    m = re.search(path+'(.+?)end', filename+'end')
    if m:
        label = m.group(1)
        if '000' in label:
            labels.append(label)

#labels = np.asarray(labels)
#np.savetxt(, labels)
thefile = open(path_saved+ filename, 'wb')
for item in labels:
  thefile.write("%s\n" % item)
thefile.close
#pickle.dump(labels, )  
