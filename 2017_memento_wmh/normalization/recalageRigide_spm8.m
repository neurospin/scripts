function recalageRigide_spm8(imageSou, imageRef, others)
% Fonction pour recaler une image source sur une image reference
%
% USAGE
%       recalageRigide_spm8(imageSou, imageRef, others)
%
% INPUTS
%       imageSou : le chemin jusqu'a l'image a recaler
%       imageRef : le chemin jusqu'a l'image de reference pour le recalage
%       others   : le chemin vers des images supplementaires a recaler
%                  (others doit etre sous forme de cellules)
%
% Author   : Fillon Ludovic
% Created  : 03/02/2012
% Modified : 16/01/2013 (Ludovic Fillon)(Modif la recup de la matrice de transfo)

% Duplication de la source car le coreg modifie son header
[pathstr, name, ext] = fileparts(char(imageSou));
system(['mkdir ', pathstr, '/tmp']);
system(['cp ', pathstr, '/', name, ext, ' ', pathstr, '/tmp/', name, ext]);

if (nargin == 2)
    others = {''};
else
    % Duplication des sources contenues dans le others car le coreg modifie
    % leur header
    for i=1:size(others,1)
        [pathstr, name, ext] = fileparts(others{i});
        system(['cp ', pathstr, '/', name, ext, ' ', pathstr, '/tmp/', name, ext ]);
    end
end

jobid = cfg_util('initjob', 'batchRecalageRigide_spm8.m');
cfg_util('filljob', jobid, imageRef, imageSou, others);
cfg_util('run', jobid);

% sauvegarde des transformations stockees dans les header.mat
% M = spm_get_space(char(imageSou));
% [pathstr, name] = fileparts(char(imageSou));
% save([pathstr, '/mat_', name, '.mat'], 'M');
% % imageT1         = ouvrir(char(imageSou));
% % matImageT1      = imageT1.header.mat;
% % [pathstr, name] = fileparts(char(imageSou));
% % save([pathstr, '/mat_', name, '.mat'], 'matImageT1');

% On ramene la T1 originale
[pathstr, name, ext] = fileparts(char(imageSou));
system(['cp -f ', pathstr, '/tmp/', name, ext, ' ', pathstr, '/', name, ext]);

% On ramene les sources dans le others
if (nargin ~= 2)
    for i=1:size(others,1)
        [pathstr, name, ext] = fileparts(others{i});
        system(['cp -f ', pathstr, '/tmp/', name, ext, ' ', pathstr, '/', name, ext]);
    end
end

system(['rm -Rf ', pathstr, '/tmp']);

end