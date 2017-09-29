function segmentationTissus_spm8 (imageT1)
% Fonction qui utilise le batchSegmentation pour extrait une carte de
% probabilite de tous les tissus a partir de la T1
%
% USAGE
%       segmentationTissus_spm8 (imageT1)
%
% INPUT
%       imageT1 : image reference pour les segmentations
%
% Author   : Fillon Ludovic
% Created  : 03/02/2012
% Modified : 03/02/2012 (Ludovic Fillon)

jobid = cfg_util('initjob', 'batchSegmentation_spm8.m');
cfg_util('filljob', jobid, imageT1);
cfg_util('run', jobid);

end