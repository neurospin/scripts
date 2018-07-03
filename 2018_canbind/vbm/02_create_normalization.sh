#!/bin/bash

echo '%-----------------------------------------------------------------------
% Job configuration created by cfg_util (rev $Rev: 6942 $)
%-----------------------------------------------------------------------' > $4

echo -ne 'matlabbatch{1}.spm.tools.dartel.mni_norm.template = {' >> $4
echo -ne \'$1\' >> $4 
echo '};' >> $4

echo '%%
matlabbatch{1}.spm.tools.dartel.mni_norm.data.subjs.flowfields = {' >> $4

echo $2 $3
flowfield_image=( `cat $2` )
gm_image=( `cat $3` )


for index in ${!flowfield_image[*]}
do
    echo \'${flowfield_image[$index]}\' >> $4
done

echo '};
%%
%%
matlabbatch{1}.spm.tools.dartel.mni_norm.data.subjs.images = {
                                                              {' >> $4

for index in ${!gm_image[*]}
do
    echo \'${gm_image[$index]}\' >> $4
done

echo "                                                              }
                                                              }';
%%
matlabbatch{1}.spm.tools.dartel.mni_norm.vox = [1 1 1];
matlabbatch{1}.spm.tools.dartel.mni_norm.bb = [NaN NaN NaN
                                               NaN NaN NaN];
matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = 1;
matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = [0 0 0];" >> $4

