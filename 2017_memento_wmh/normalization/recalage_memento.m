% Recalage des masques WHASA vers l'espace T1 puis MNI.
%
% USAGE
%       recalage_memento (listT1, listFLAIR, listMask)
%
% INPUTS
%       listT1    : liste des images T1 (raw)
%       listFLAIR : liste des images FLAIR corrigées du biais
%       listMask  : liste des masques de segmentation WHASA
%
% Author   : Fillon Ludovic
% Created  : 20/07/2017
% Modified : 23/08/2017

% listes
listT1    = liste_fichiers('/aramis/dartagnan2/fillon/Memento_registration/3DT1', '0*.nii');
listFLAIR = liste_fichiers('/aramis/dartagnan2/fillon/Memento_registration/T2FLAIR', 'nobias*.nii');
listWMH   = liste_fichiers('/aramis/dartagnan2/fillon/Memento_registration/T2FLAIR', 'wmh*.nii');

for i=1:size(listT1, 1)
    % correction de biais
    segmentationTissus_spm8(listT1(i));
    
    % recalage rigide
    [pathstr, name, ext] = fileparts(char(listT1(i)));
    pattern1 = ['m', name, '.nii'];
    listmT1  = liste_fichiers('/aramis/dartagnan2/fillon/Memento_registration/3DT1', pattern1);
    recalageRigide_spm8(listFLAIR(i), listmT1(1), listWMH(i));
    
    % deformation
    pattern2 = ['rwmh_lesion_mask_', name, '.nii'];
    listrWMH = liste_fichiers('/aramis/dartagnan2/fillon/Memento_registration/T2FLAIR', pattern2);
    pattern3 = ['iy_', name, '.nii'];
    listdef  = liste_fichiers('/aramis/dartagnan2/fillon/Memento_registration/3DT1', pattern3);
    native2MNI_spm8( listrWMH(1), listdef(1) );
end
