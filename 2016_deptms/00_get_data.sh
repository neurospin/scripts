# Copy T1 images
# ==============

#cp /volatile/duchesnay/data/11Deprim.tgz /neurospin/brainomics/2016_deptms/data_original/
#cd /volatile/duchesnay/data/

WD="/neurospin/brainomics/2016_deptms/data_original"
cd $WD
tar xzvf 11Deprim.tgz

cd 11Deprim/data/mri
ls */*.img|wc
#     34      34     572
output="$WD/t1"
ls */*.img|while read input_img ; do 
        subject=$(dirname $input_img)
        #output_dirname="$output/$subject"
        output_dirname=$output
        tmp=$(basename $input_img)
        input_basename="${tmp%.img}"
        #mkdir ${output_dirname}
        cp $subject/$subject.* ${output_dirname}
        fsl5.0-fslchfiletype NIFTI "${output_dirname}/$subject.img"
        #$input_basename;
done

ls ${output}/*.nii|wc
     34      34    2204


# Clinical data (Python)
# ======================


## Read pop csv
import os.path
import pandas as pd

INPUT='/neurospin/brainomics/2014_deptms/base_data/clinic/deprimPetInfo.csv'
OUTPUT='/neurospin/brainomics/2016_deptms/data_original'

data = pd.read_csv(INPUT, sep='\t')

data["subject"] = [f.replace('smwc1', '').replace('.img', '') for f in data.MRIm_G_file]

np.all([os.path.exists(os.path.join(OUTPUT, s, s+".img")) for s in data.subject])
data.to_csv(os.path.join(OUTPUT, 'deptms_info.csv'), index=False)


## Convert to niftii
#WD="/neurospin/brainomics/2016_deptms/data_original"
#cd $WD
#ls *.img|while read input_img ; do fsl5.0-fslchfiletype NIFTI $input_img ; done

