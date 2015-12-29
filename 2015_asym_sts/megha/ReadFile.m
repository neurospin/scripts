function [Pheno, Cov, K, Nsubj, Npheno, Ncov, PhenoNames, SubjID] = ReadFile(PhenoFile, header, CovFile, delimiter, GRMFile, GRMid)
% This function parses phenotypic data and covariates, and constructs genetic relationship matrix (GRM).
% Subjects in these files do not need to be exactly the same.
% This function will find the subjects in common in these files and sort the order of the subjects.
%
% Input arguments -
% PhenoFile: a plain text file containing phenotypic data
% Col 1: family ID; Col 2: subject ID; From Col 3: numerical values
% header: 1 - PhenoFile contains a headerline
% 0 - PhenoFile does not contain a headerline
% CovFile: a plain text file containing covariates (intercept NOT included)
% If no covariates need to be included in the model, set CovFile = '';
% Col 1: family ID; Col 2: subject ID; From Col 3: numerical values
% delimiter: delimiter used in the phenotypic file and covariates file
% GRMFile: a plain text file containing the lower triangle elements of the GRM
% Col 1 & 2: indices of pairs of individuals; 
% Col 3: number of non-missing SNPs; Col 4: the estimate of genetic relatedness 
% GRMid: a plain text file of subject IDs corresponding to GRMFile
% Col 1: family ID; Col 2: subject ID
%
% Output arguments -
% Pheno: an Nsubj x Npheno phenotypic data matrix
% Cov: an Nsubj x Ncov covariate data matrix
% K: an Nsubj x Nsubj GRM
% Nsubj: total number of subjects in common
% Npheno: number of phenotypes
% Ncov: number of covariates (including intercept)
% PhenoNames: names of the phenotypes in the headerline
% SubjID: a list of subject IDs

[NsubjP, Npheno, SubjIDP, Pheno, PhenoNames] = ParseFile(PhenoFile, delimiter, header);   % parse the phenotypic data
if ~isempty(CovFile)
	[NsubjC, Ncov, SubjIDC, Cov] = ParseFile(CovFile, delimiter);   % parse the covariates file
end
[NsubjG, SubjIDG, K] = ParseGRM(GRMFile, GRMid);   % parse the GRM files

if isempty(CovFile)
    SubjID = intersect(SubjIDP, SubjIDG);   % find subjects in common
else
    SubjID = intersect(intersect(SubjIDP, SubjIDC), SubjIDG);   % find subjects in common
end
Nsubj = length(SubjID);   % calculate the total number of common subjects

% sort the order of the subjects in the phenotypic data
IdxP = zeros(Nsubj,1);
for i = 1:Nsubj
    for j = 1:NsubjP
        if strcmp(SubjID{i}, SubjIDP{j})
            IdxP(i) = j;
            break
        end
    end
end
Pheno = Pheno(IdxP,:);

% sort the order of the subjects in the covariate file
if isempty(CovFile)   % no covariates included in the model
    Cov = ones(Nsubj,1);   % intercept
    Ncov = 1;   % number of covariates (intercept only)
else
    IdxC = zeros(Nsubj,1);
    for i = 1:Nsubj
        for j = 1:NsubjC
            if strcmp(SubjID{i}, SubjIDC{j})
                IdxC(i) = j;
                break
            end
        end
    end
    Cov = [ones(Nsubj,1), Cov(IdxC,:)];   % add intercept
    Ncov = Ncov+1;   % number of covariates (with intercept)
end

% sort the order of the subjects in the GRM
IdxG = zeros(Nsubj,1);
for i = 1:Nsubj
    for j = 1:NsubjG
        if strcmp(SubjID{i}, SubjIDG{j})
            IdxG(i) = j;
            break
        end
    end
end
K = K(IdxG,IdxG);
%