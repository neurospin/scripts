function [SurfLh, SurfRh, Cov, K, Nsubj, NvetLh, NvetRh, Ncov, SubjID] = ReadFileSurf(SurfDir, ImgSubj, ImgFileLh, ImgFileRh, FSDir, CovFile, delimiter, GRMFile, GRMid)
% This function reads surface data and covariates, and constructs genetic relationship matrix (GRM).
% Subjects in these files do not need to be exactly the same.
% This function will find the subjects in common in these files and sort the order of the subjects.
%
% Input arguments -
% SurfDir: directory of the surface data (SUBJECTS_DIR)
% SubjID: a plain text file containing a list of subject IDs with imaging data to be included in the analysis
% ImgFileLh/ImgFileRh: name of the file containing surface data for the left/right hemisphere
% FSDir: directory of FreeSurfer where the folder "subjects" can be found
% CovFile: a plain text file containing covariates (intercept NOT included)
% If no covariates need to be included in the model, set CovFile = '';
% Col 1: family ID; Col 2: subject ID; From Col 3: numerical values
% delimiter: delimiter used in the covariates file
% GRMFile: a plain text file containing the lower triangle elements of the GRM
% Col 1 & 2: indices of pairs of individuals; 
% Col 3: number of non-missing SNPs; Col 4: the estimate of genetic relatedness 
% GRMid: a plain text file of subject IDs corresponding to GRMFile
% Col 1: family ID; Col 2: subject ID
%
% Output arguments -
% SurfLh/SurfRh: an Nsubj x NvetLh/NvetRh data matrix for the left/right hemisphere
% Cov: an Nsubj x Ncov covariate data matrix
% K: an Nsubj x Nsubj GRM
% Nsubj: total number of subjects in common
% NvetLh/NvetRh: number of in-mask vertices on the left/right hemisphere
% Ncov: number of covariates (including intercept)
% SubjID: a list of subject IDs

if ~isempty(CovFile)
    [NsubjC, Ncov, SubjIDC, Cov] = ParseFile(CovFile, delimiter);   % parse the covariates file
end
[NsubjG, SubjIDG, K] = ParseGRM(GRMFile, GRMid);   % parse the GRM files

fid = fopen(ImgSubj);   % open the file containing subject IDs for imaging data
SubjIDP = textscan(fid,'%s');   % read subject IDs
SubjIDP = SubjIDP{1};   % convert a 1x1 cell structure into a Nsubj x 1 cell structure

if isempty(CovFile)
    SubjID = intersect(SubjIDP, SubjIDG);   % find subjects in common
else
    SubjID = intersect(intersect(SubjIDP, SubjIDC), SubjIDG);   % find subjects in common
end
Nsubj = length(SubjID);   % total number of common subjects

disp('----- Extract Surface Data -----')
[NvetLh, NvetRh, SurfLh, SurfRh] = ExtractSurf(SurfDir, SubjID, ImgFileLh, ImgFileRh, FSDir);   % extract surface data

% sort the order of the subjects in the covariate file
if isempty(CovFile)   % no covariates included in the model
    Cov = ones(Nsubj,1);   % intercept
    Ncov = 1;
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
    Ncov = Ncov+1;   % add intercept
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