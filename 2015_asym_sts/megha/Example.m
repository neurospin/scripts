%% Examples

%% MEGHA.m
% I have included in this file randperm(...,'Seed',0) to fixe the permutation 
directory_pheno = '/volatile/yann/2015_asym_sts/PLINK_all_pheno0.02/';
pheno_files = dir(strcat(directory_pheno,'*.phe'));
tic;
PhenoFile = '/volatile/yann/2015_asym_sts/PLINK_all_pheno0.02/concatenated_pheno.phe';%'/neurospin/brainomics/2015_asym_sts/pheno/STs.phe';
header = 1;
CovFile = '/volatile/yann/imagen_central/covar/MEGHA_covar.cov';%/neurospin/brainomics/imagen_central/covar/covar_GenCitHan_MEGHA.cov';
delimiter = '\t';
GRMFile = '/volatile/yann/imagen_central/kinship/prunedYann_m0.01_g1_h6_wsi50_wsk5_vif10.0.grm';
GRMid = '/volatile/yann/imagen_central/kinship/prunedYann_m0.01_g1_h6_wsi50_wsk5_vif10.0.grm.id';
Nperm = 1000000;
WriteStat = 1;
OutDir = strcat('/volatile/yann/megha/all_sulci_',int2str(Nperm),'perm_fullcovar/all_pheno_without_asym_',int2str(Nperm),'perm'); 
%
[Pval, h2, SE, PermPval, PermFWEcPval, Nsubj, Npheno, Ncov] = MEGHA(PhenoFile, header, CovFile, delimiter, GRMFile, GRMid, Nperm, WriteStat, OutDir);
toc
%% MEGHAmat.m
% load Pheno; load Cov; load K;
% Nperm = 1000;
% %
% [Pval, h2, SE, PermPval, PermFWEcPval, Nsubj, Npheno, Ncov] = MEGHAmat(Pheno, Cov, K, Nperm);
%% MEGHAsurf.m
% SurfDir = '/path/SUBJECT_DIR/';
% ImgSubj = 'path/ImgID.txt';
% ImgFileLh = 'lh.fsaverage.thickness.fwhm20.mgh';
% ImgFileRh = 'rh.fsaverage.thickness.fwhm20.mgh';
% FSDir = '/path/freesurfer/';
% CovFile = '/path/qcovar.txt';
% delimiter = '\t';
% GRMFile = '/path/GRM.grm';
% GRMid = '/path/GRM.grm.id';
% WriteImg = 1;
% OutDir = '/OutPutDirectory/';
% Nperm = 1000;
% Pthre = 0.01;
% %
% [PvalLh, PvalRh, h2Lh, h2Rh, SE, ClusPLh, ClusPRh, PeakLh, ClusLh, ClusidLh, PeakRh, ClusRh, ClusidRh, Nsubj, NvetLh, NvetRh, Ncov] = ...
%     MEGHASurf(SurfDir, ImgSubj, ImgFileLh, ImgFileRh, FSDir, CovFile, delimiter, GRMFile, GRMid, WriteImg, OutDir, Nperm, Pthre);
%% MEGHAsurfmat.m
% load PhenoSurf; load Cov; load K;
% FSDir = '/path/freesurfer/';
% Nperm = 1000;
% Pthre = 0.01;
% %
% [PvalLh, PvalRh, h2Lh, h2Rh, SE,  ClusPLh, ClusPRh, PeakLh, ClusLh, ClusidLh, PeakRh, ClusRh, ClusidRh, Nsubj, NvetLh, NvetRh, Ncov] = ...
%     MEGHASurfmat(FSDir, SurfLh, SurfRh, Cov, K, Nperm, Pthre);
%%