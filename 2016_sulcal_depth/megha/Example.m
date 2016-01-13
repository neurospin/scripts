%% Examples

clear 
clc
disp('----- Do not forget to set pathdef to include surfstat -----')
addpath('/neurospin/brainomics/surfstat'))

tic
%% MEGHAsurf.m
for j= 1:2
SurfDir = '/neurospin/imagen/BL/processed/freesurfer/smooth_textures/';
ImgSubj = strcat(pwd,'/labels.txt');

ImgFileLh = '_lh.curv.pial.mgz';
ImgFileRh = '_rh.curv.pial.mgz';
FSDir = '/neurospin/imagen/BL/processed/freesurfer/';

if j == 1
    CoVFileBasename = 'covar_GenCit5PCA_ICV_MEGHA';
elseif j == 2
    CoVFileBasename = 'covar_GenCitHan5PCA_ICV_MEGHA';
end

CovFile = strcat('/neurospin/brainomics/imagen_central/clean_covar/',CoVFileBasename, '.cov');%/neurospin/brainomics/imagen_central/covar/covar_GenCitHan_MEGHA.cov';
delimiter = '\t';
GRMFile = '/neurospin/brainomics/imagen_central/kinship/prunedYann_m0.01_g1_h6_wsi50_wsk5_vif10.0.grm';
GRMid = '/neurospin/brainomics/imagen_central/kinship/prunedYann_m0.01_g1_h6_wsi50_wsk5_vif10.0.grm.id';
Nperm = 1000;
WriteImg = 1;
OutDir = strcat(pwd,'/smooth_curv_pial/', CoVFileBasename);

Pthre = 0.01;
%
[PvalLh, PvalRh, h2Lh, h2Rh, SE, ClusPLh, ClusPRh, PeakLh, ClusLh, ClusidLh, PeakRh, ClusRh, ClusidRh, Nsubj, NvetLh, NvetRh, Ncov] = ...
    MEGHASurf(SurfDir, ImgSubj, ImgFileLh, ImgFileRh, FSDir, CovFile, delimiter, GRMFile, GRMid, WriteImg, OutDir, Nperm, Pthre);
save(strcat(OutDir,CoVFileBasename,'.mat'))
end
toc
%% MEGHAsurfmat.m
% load PhenoSurf; load Cov; load K;
% FSDir = '/path/freesurfer/';
% Nperm = 1000;
% Pthre = 0.01;
% %
% [PvalLh, PvalRh, h2Lh, h2Rh, SE,  ClusPLh, ClusPRh, PeakLh, ClusLh, ClusidLh, PeakRh, ClusRh, ClusidRh, Nsubj, NvetLh, NvetRh, Ncov] = ...
%     MEGHASurfmat(FSDir, SurfLh, SurfRh, Cov, K, Nperm, Pthre);
%%
