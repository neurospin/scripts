function WriteStats(OutDir, Npheno, Nperm, header, PhenoNames, Pval, h2, PermPval, PermFWEcPval)
% This function writes MEGHA significance measures and point estimates of heritability to the output directory
%
% Input arguments -
% OutDir: output directory; default directory is the working directory
% Npheno: number of phenotypes
% Nperm: number of permutations
% header: 1 - PhenoFile contains a headerline
% 0 - PhenoFile does not contain a headerline
% PhenoNames: names of the phenotypes in the headerline
% Pval: MEGHA p-values
% h2: MAGHA estimates of heritability magnitude
% PermPval: MEGHA permutation p-values
% PermFWEcPval: MEGHA family-wise error corrected permutation p-values
%
% Output files -
% MEGHAstat.txt: point estimates and significance measures of heritability
% for each phenotype written to the output directory "OutDir"

if strcmp(OutDir, 'NA')
    fid = fopen('MEGHAstat.txt', 'w');   % open the file
else
    fid = fopen([OutDir, 'MEGHAstat.txt'], 'w');   % open the file
end

if header == 1
    if Nperm > 0
        fprintf(fid, '%s\t%s\t%s\t%s\t%s\n', 'Phenotype', 'h2', 'Pval', 'PermPval', 'PermFWEcPval');   % write headerline
        for i = 1:Npheno
            fprintf(fid, '%s\t%f\t%f\t%f\t%f\n', cell2mat(PhenoNames{i}), h2(i), Pval(i), PermPval(i), PermFWEcPval(i));   % write statistics
        end
    else   % no permutation statistics
        fprintf(fid, '%s\t%s\t%s\n', 'Phenotype', 'h2', 'Pval');   % write headerline
        for i = 1:Npheno
            fprintf(fid, '%s\t%f\t%f\n', cell2mat(PhenoNames{i}), h2(i), Pval(i));
        end
    end
else   % no headerline
    if Nperm > 0
        fprintf(fid, '%s\t%s\t%s\t%s\n', 'h2', 'Pval', 'PermPval', 'PermFWEcPval');   % write headerline
        for i = 1:Npheno
            fprintf(fid, '%f\t%f\t%f\t%f\n', h2(i), Pval(i), PermPval(i), PermFWEcPval(i));   % write statistics
        end
    else   % no permutation statistics
        fprintf(fid, '%s\t%s\n', 'h2', 'Pval');   % write headerline
        for i = 1:Npheno
            fprintf(fid, '%f\t%f\n', h2(i), Pval(i));   % write statistics
        end
    end
end

fclose(fid);   % close the file
%