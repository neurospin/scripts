%-----------------------------------------------------------------------
% Job configuration created by cfg_util (rev $Rev: 4252 $)
%-----------------------------------------------------------------------
matlabbatch{1}.spm.spatial.coreg.estwrite.ref = '<UNDEFINED>';
matlabbatch{1}.spm.spatial.coreg.estwrite.source = '<UNDEFINED>';
matlabbatch{1}.spm.spatial.coreg.estwrite.other = '<UNDEFINED>';
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi'; %ncc
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.interp = 1; % 0:for nearest neighbour 1:trilinear
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.mask = 0;
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';

%matlabbatch{1}.spm.spatial.coreg.estwrite.other = {''};
