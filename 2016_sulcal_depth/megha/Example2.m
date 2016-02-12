%% Examples

clear 
clc
disp('----- Do not forget to set pathdef to include surfstat -----')
addpath('/neurospin/brainomics/2016_sulcal_depth/surfstat')

%% MEGHA.m
%I have included in this file randperm(...,'Seed',0) to fix the permutation 

features = ['surface     '; 'opening     '; 'GM_thickness'; 'length      '; 'depthMean   '; 'depthMax    ';];
feature = 'depthMax';
for i=1:1
for j=1:1
tic

if j == 1
    CoVFileBasename = 'covar_Gen5PCA_ICV_MEGHA';
    %CoVFileBasename = 'ICV';
elseif j == 2
    CoVFileBasename = 'covar_GenCitHan5PCA_ICV_MEGHA';
end

filename = 'concatenated_pheno';

%PhenoFile = strcat('/neurospin/brainomics/2016_sulcal_depth/pheno/all_features/', char(features(i,:)), '0.02/all_sulci_qc/',filename,'.phe');
PhenoFile = strcat('/neurospin/brainomics/2016_connectogram/normalized_connecto_pheno_reduced.phe');
header = 1;
CovFile = strcat('/neurospin/brainomics/imagen_central/clean_covar/', CoVFileBasename,'.cov');
delimiter = '\t';
GRMFile = '/neurospin/brainomics/imagen_central/kinship/prunedYann_m0.01_g1_h6_wsi50_wsk5_vif10.0.grm';
GRMid = '/neurospin/brainomics/imagen_central/kinship/prunedYann_m0.01_g1_h6_wsi50_wsk5_vif10.0.grm.id';
Nperm = 100;
WriteStat = 1;
%out_directory =  strcat('/neurospin/brainomics/2016_sulcal_depth/megha/all_features/', char(features(i,:)), 'tol0.02/');
out_directory =  strcat('/neurospin/brainomics/2016_connectogram/');
if exist(out_directory)
else
mkdir(out_directory)    
end
out_filename = strcat('100reduced',CoVFileBasename);
OutDir = strcat(out_directory,out_filename); 
%
[Pval, h2, SE, PermPval, PermFWEcPval, Nsubj, Npheno, Ncov] = MEGHA(PhenoFile, header, CovFile, delimiter, GRMFile, GRMid, Nperm, WriteStat, OutDir);
toc
end
end
%% MEGHAmat.m
% load Pheno; load Cov; load K;
% Nperm = 1000;
% %
% [Pval, h2, SE, PermPval, PermFWEcPval, Nsubj, Npheno, Ncov] = MEGHAmat(Pheno, Cov, K, Nperm);
