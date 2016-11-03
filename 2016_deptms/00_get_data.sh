# Copy T1 images
# ==============

cp /volatile/duchesnay/data/11Deprim.tgz /neurospin/brainomics/2016_deptms/data_original/
cd /volatile/duchesnay/data/
tar xzvf 11Deprim.tgz

cd 11Deprim/data/mri
ls */*.img|wc
     34      34     572
output="/neurospin/brainomics/2016_deptms/data_original"
ls */*.img|while read input_img ; do 
        subject=$(dirname $input_img)
        output_dirname="$output/$subject"
        tmp=$(basename $input_img)
        input_basename="${tmp%.img}"
        mkdir ${output_dirname}
        cp $subject/$subject.* ${output_dirname}
        #$input_basename;
done

ls ${output}/*/*.img|wc
     34      34    2204


# Clinical data (Python)
# ======================


## Read pop csv
import os.path
import pandas as pd

INPUT='/neurospin/brainomics/2014_deptms/base_data/clinic/deprimPetInfo.csv'
OUTPUT='/neurospin/brainomics/2016_deptms/data_original'
pop = pd.read_csv(INPUT, sep="\t")

data = pd.read_csv(INPUT, sep='\t')

data["subject"] = [f.replace('smwc1', '').replace('.img', '') for f in data.MRIm_G_file]

np.all([os.path.exists(os.path.join(OUTPUT, s, s+".img")) for s in data.subject])
data.to_csv(os.path.join(OUTPUT, 'deptms_info.csv'), index=False)

