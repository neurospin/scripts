#!/bin/bash

echo '%-----------------------------------------------------------------------
% Job configuration created by cfg_util (rev $Rev: 4252 $)
%-----------------------------------------------------------------------


matlabbatch{1}.spm.tools.dartel.mni_norm.data.subjs.flowfields = {
                                               {' > $3
flowfield_image=( `cat $1` )
gm_image=( `cat $2` )

for index in ${!flowfield_image[*]}
do
    echo \'${flowfield_image[$index]}\' >> $3 
done

echo                                               }}';



matlabbatch{1}.spm.tools.dartel.mni_norm.data.subjs.images = {
                                               {' >> $3

for index in ${!gm_image[*]}
do
    echo \'${gm_image[$index]}\' >> $3 
done

echo "                                               }}';


matlabbatch{1}.spm.tools.dartel.mni_norm.template = {'/neurospin/brainomics/2018_euaims_leap_predict_vbm/data/processed/LEAP_V01/100693509718/anatomy/Template_6.nii'}; 


matlabbatch{1}.spm.tools.dartel.mni_norm.vox = [1 1 1];
matlabbatch{1}.spm.tools.dartel.mni_norm.bb = [NaN NaN NaN
                                               NaN NaN NaN];
matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = 1;
matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = [0 0 0];" >> $3
