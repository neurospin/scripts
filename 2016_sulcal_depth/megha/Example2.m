%% Examples

clear 
clc
disp('----- Do not forget to set pathdef to include surfstat -----')
addpath('/neurospin/brainomics/surfstat')

%% MEGHA.m
%I have included in this file randperm(...,'Seed',0) to fix the permutation 
for j=1:2
tic
if j == 1
    CoVFileBasename = 'covar_GenCit5PCA_ICV_MEGHA';
elseif j == 2
    CoVFileBasename = 'covar_GenCitHan5PCA_ICV_MEGHA';
end

filename = 'concatenated_pheno';

PhenoFile = strcat('/neurospin/brainomics/2016_sulcal_depth/pheno/PLINK_all_pheno0.02/all_sulci_qc/',filename,'.phe');
header = 1;
CovFile = strcat('/neurospin/brainomics/imagen_central/clean_covar/', CoVFileBasename,'.cov');
delimiter = '\t';
GRMFile = '/neurospin/brainomics/imagen_central/kinship/prunedYann_m0.01_g1_h6_wsi50_wsk5_vif10.0.grm';
GRMid = '/neurospin/brainomics/imagen_central/kinship/prunedYann_m0.01_g1_h6_wsi50_wsk5_vif10.0.grm.id';
Nperm = 10;
WriteStat = 1;
out_directory =  '/neurospin/brainomics/2016_sulcal_depth/megha/all_sulci_qc/tol0.02/';
if exist(out_directory)
else
mkdir(out_directory)    
end
out_filename = CoVFileBasename;
OutDir = strcat(out_directory,out_filename); 
%
[Pval, h2, SE, PermPval, PermFWEcPval, Nsubj, Npheno, Ncov] = MEGHA(PhenoFile, header, CovFile, delimiter, GRMFile, GRMid, Nperm, WriteStat, OutDir);
toc
end
%% MEGHAmat.m
% load Pheno; load Cov; load K;
% Nperm = 1000;
% %
% [Pval, h2, SE, PermPval, PermFWEcPval, Nsubj, Npheno, Ncov] = MEGHAmat(Pheno, Cov, K, Nperm);
