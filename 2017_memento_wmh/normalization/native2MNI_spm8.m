function native2MNI_spm8( image_to_deform, deformation_field )
% Fonction pour deformer des images de l'espace T1 vers l'espace normalise
% MNI a partir des champs de deformations calcules avec new segment
%
% USAGE
%       native2MNI_spm8( image_to_deform, deformation_field )
%
% INPUT
%       image_to_deform   : image a deformer vers l'espace MNI
%       deformation_field : image des champs de deformation direct (y)
%
% Author   : Fillon Ludovic
% Created  : 25/11/2015
% Modified : 25/11/2015 (Ludovic Fillon)

[pathstr, name] = fileparts(char(deformation_field));

jobid    = cfg_util('initjob', 'batchDeformationField_spm8.m');
cfg_util('filljob', jobid, deformation_field, image_to_deform, {pathstr});
%cfg_util('savejob', jobid, '~/tmp/test.m');
cfg_util('run', jobid);

system(['rm -f ', pathstr, '/y_deformation.nii ']);

end
