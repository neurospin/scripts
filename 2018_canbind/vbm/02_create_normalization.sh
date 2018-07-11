#!/bin/bash

echo '%-----------------------------------------------------------------------
% Job configuration created by cfg_util (rev $Rev: 6942 $)
%-----------------------------------------------------------------------' > $5

echo -ne 'matlabbatch{1}.spm.tools.dartel.mni_norm.template = {' >> $5
echo -ne \'$1\' >> $5 
echo '};' >> $5

echo '%%
matlabbatch{1}.spm.tools.dartel.mni_norm.data.subjs.flowfields = {' >> $5

echo $2 $3
flowfield_image=( `cat $2` )
gm_image=( `cat $3` )


for index in ${!flowfield_image[*]}
do
    echo \'${flowfield_image[$index]}\' >> $5
done

echo '};
%%
%%
matlabbatch{1}.spm.tools.dartel.mni_norm.data.subjs.images = {
                                                              {' >> $5

for index in ${!gm_image[*]}
do
    echo \'${gm_image[$index]}\' >> $5
done

echo "                                                              }
                                                              }';
%%
matlabbatch{1}.spm.tools.dartel.mni_norm.vox = [${4} ${4} ${4}];
matlabbatch{1}.spm.tools.dartel.mni_norm.bb = [NaN NaN NaN
                                               NaN NaN NaN];
matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = 1;
matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = [0 0 0];" >> $5

