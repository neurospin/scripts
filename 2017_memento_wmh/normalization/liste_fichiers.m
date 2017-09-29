function [liste]=liste_fichiers(repertoire_racine, caractere_cibles)

os=computer;
delimiter=sprintf('\n');

if strcmp(os,'PCWIN') || strcmp(os,'PCWIN64')
    [trash,chaine]=system(sprintf('dir /S /B "%s"',fullfile(repertoire_racine,caractere_cibles)));
    
else
    [trash,chaine]=system(sprintf('find %s -name "%s"',repertoire_racine,caractere_cibles));
end

liste=[];
while ~isempty(deblank(chaine))
    tmp=strtok(chaine,delimiter);
    chaine=strrep(chaine,tmp,'');
    liste=strvcat(liste,tmp);
end

if ~isempty(liste)
    liste=cellstr(liste);
    liste=sort(liste);
end
