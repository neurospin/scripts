function [PvalLh, PvalRh, h2Lh, h2Rh, SE,  ClusPLh, ClusPRh, PeakLh, ClusLh, ClusidLh, PeakRh, ClusRh, ClusidRh, Nsubj, NvetLh, NvetRh, Ncov] = ...
    MEGHASurfmat(FSDir, SurfLh, SurfRh, Cov, K, Nperm, Pthre)
% This function implements massively expedited genome-wide heritability analysis (MEGHA) for surface data resampled on FreeSurfer's fsaverage 
% when surface data, covariates and GRM have been prepared in Matlab .mat format.
% Need FreeSurfer MATLAB tools and the MATLAB toolbox surfstat developed by Prof. Keith J. Worsley 
% to read surface data and perform surface based clustering.
% Software download: http://www.math.mcgill.ca/keith/surfstat/
%
% Input arguments -
% FSDir: directory of FreeSurfer where the folder "subjects" can be found (FREESURFER_DIR)
% SurfLh/SurfRh: an Nsubj x NvetLh/NvetRh data matrix for the left/right hemisphere
% Cov: an Nsubj x Ncov covariates matrix (intercept included)
% K: an Nsubj x Nsubj GRM
%
% [Note: Subjects in SurfLh/SurfRh, Cov and K must be exactly the same and arranged in the same order.]
%
% Nperm: number of permutations for cluster inference; set Nperm = 0 if permutation inference is not needed; default Nperm = 0
% Pthre: p-value threshold for permutation based cluster inference; default Pthre = 0.01
%
% Output arguments -
% PvalLh/PvalRh: MEGHA p-values for in-mask vertices on the left/right hemisphere
% h2Lh/h2Rh: MEGHA heritability estimates for in-mask vertices on the left/right hemisphere
% SE: estimated standard error of heritability magnitude
% ClusPLh/ClusPRh: family-wise error corrected p-values for clusters above the threshold 
% on the left/right hemisphere obtained by permutation based cluster inference
% PeakLh.t/PeakRh.t: a vector of peaks (local maxima) above the threshold on the left/right hemisphere
% PeakLh.vertid/PeakRh.vertid: vertex IDs (1-based) for the peaks above the threshold on the left/right hemisphere
% PeakLh.clusid/PeakRh.clusid: cluster IDs that contain the peak on the left/right hemisphere
% ClusLh.clusid/ClusRh.clusid: cluster IDs on the left/right hemisphere
% ClusLh.nverts/ClusRh.nverts: number of vertices in each cluster on the left/right hemisphere
% ClusLh.resels/ClusRh.resels: resels in each cluster on the left/right hemisphere
% ClusidLh/ClusidRh: cluster IDs for each vertex on the left/right hemisphere
% Nsubj: total number of subjects
% NvetLh/NvetRh: number of in-mask vertices on the left/right hemisphere
% Ncov: number of covariates (including intercept)

% check inputs
if nargin < 5
    error('Not enough input arguments')
elseif nargin == 5
    Nperm = 0; Pthre = 0.01;
elseif nargin == 6
    Pthre = 0.01;
elseif nargin > 7
    error('Too many input arguments')
end

disp('----- Parse Data Files -----')
[~, NvetLh] = size(SurfLh); [~, NvetRh] = size(SurfRh);   % calculate and the number of vertices
[Nsubj, Ncov] = size(Cov);   % calculate the number of subjects and the number of covariates

disp(['----- ', num2str(Nsubj), ' Subjects, ', num2str(NvetLh), ' Verties on the left hemisphere, ', ...
    num2str(NvetRh), ' Verties on the right hemisphere, ', num2str(Ncov), ' Covariates -----'])

disp('----- Compute Surface Map -----')
P0 = eye(Nsubj) - Cov*((Cov'*Cov)\Cov');   % compute the projection matrix

% Satterthwaite approximation
delta = trace(P0*K)/2;
I11 = trace(P0*K*P0*K)/2; I12 = trace(P0*K*P0)/2; I22 = trace(P0*P0)/2;
I = I11-I12^2/I22;   % efficient information
kappa = I/(2*delta); nv = 2*delta^2/I;   % compute the scaling parameter and the degrees of freedom

ErrLh = P0*SurfLh; ErrRh = P0*SurfRh;   % compute the residuals
Sigma0Lh = sum(ErrLh.^2)'/(Nsubj-Ncov); Sigma0Rh = sum(ErrRh.^2)'/(Nsubj-Ncov);   % estimate of residual variance

Nunit = 100; NblkLh = ceil(NvetLh/Nunit); NblkRh = ceil(NvetRh/Nunit);   % the size of block and total number of blocks 
StatLh = zeros(1,NvetLh); StatRh = zeros(1,NvetRh);   % allocate space

% compute score test statistics
for i = 1:NblkLh
    if i ~= NblkLh
        StatLh((i-1)*Nunit+1:i*Nunit) = diag(ErrLh(:,(i-1)*Nunit+1:i*Nunit)'*K*ErrLh(:,(i-1)*Nunit+1:i*Nunit))./(2*Sigma0Lh((i-1)*Nunit+1:i*Nunit));
    else
        StatLh((i-1)*Nunit+1:end) = diag(ErrLh(:,(i-1)*Nunit+1:end)'*K*ErrLh(:,(i-1)*Nunit+1:end))./(2*Sigma0Lh((i-1)*Nunit+1:end));
    end
end

for i = 1:NblkRh
    if i ~= NblkRh
        StatRh((i-1)*Nunit+1:i*Nunit) = diag(ErrRh(:,(i-1)*Nunit+1:i*Nunit)'*K*ErrRh(:,(i-1)*Nunit+1:i*Nunit))./(2*Sigma0Rh((i-1)*Nunit+1:i*Nunit));
    else
        StatRh((i-1)*Nunit+1:end) = diag(ErrRh(:,(i-1)*Nunit+1:end)'*K*ErrRh(:,(i-1)*Nunit+1:end))./(2*Sigma0Rh((i-1)*Nunit+1:end));
    end
end

PvalLh = 1-chi2cdf(StatLh/kappa,nv); PvalRh = 1-chi2cdf(StatRh/kappa,nv);    % compute MEGHA p-values

[U,~,~] = svd(P0); U = U(:,1:Nsubj-Ncov);   % singular value decomposition
TK = U'*K*U;   % transformed GRM
TNsubj = Nsubj-Ncov;   % transformed number of subjects
stdK = std(TK(triu(true(TNsubj),1)));   % standard deviation of the off-diagonal elements of the transformed GRM

% compute estimates of heritability magnitude
WaldStatLh = chi2inv(max(1-2*PvalLh,0), 1); WaldStatRh = chi2inv(max(1-2*PvalRh,0), 1);   % Wald statistic
% SE = 316/Nsubj;   % theoretical estimate of the standard error of heritability magnitude
SE = sqrt(2)/TNsubj/stdK;   % empirical estimate of the standard error of heritability magnitude
h2Lh = min(SE*sqrt(WaldStatLh), 1); h2Rh = min(SE*sqrt(WaldStatRh), 1);   % heritability estimates

disp('----- Surface-based Clustering -----')
% read mask files
fid = fopen([FSDir, 'subjects/fsaverage/label/lh.cortex.label']);   % open mask file for the lh
LabelLh = textscan(fid,'%u %f %f %f %f','Headerlines',2);   % skip the first and second line and read the rest of the file
MaskLh = LabelLh{1}+1;   % convert 0-based index to 1-based

fid = fopen([FSDir, 'subjects/fsaverage/label/rh.cortex.label']);   % open mask file for the rh
LabelRh = textscan(fid,'%u %f %f %f %f','Headerlines',2);   % skip the first and second line and read the rest of the file
MaskRh = LabelRh{1}+1;   % convert 0-based index to 1-based

% read triangular system
TempLh = [FSDir 'subjects/fsaverage/surf/lh.sphere.reg'];
TriLh = SurfStatReadSurf(TempLh,'b');
NvetTotLh = size(TriLh.coord,2);   % total number of vertices

TempRh = [FSDir 'subjects/fsaverage/surf/rh.sphere.reg'];
TriRh = SurfStatReadSurf(TempRh,'b');
NvetTotRh = size(TriRh.coord,2);

% create binary masks
BinMaskLh = zeros(1,NvetTotLh); BinMaskLh(MaskLh) = 1;
BinMaskRh = zeros(1,NvetTotRh); BinMaskRh(MaskRh) = 1;

% surface based clustering
Thre = chi2inv(1-Pthre,nv)*kappa;   % threshold of test statistic

slmLh.t = zeros(1,NvetTotLh); slmLh.t(MaskLh) = StatLh;
slmRh.t = zeros(1,NvetTotRh); slmRh.t(MaskRh) = StatRh;
slmLh.tri = TriLh.tri; slmRh.tri = TriRh.tri;

[PeakLh, ClusLh, ClusidLh] = SurfStatPeakClus(slmLh, BinMaskLh, Thre);
[PeakRh, ClusRh, ClusidRh] = SurfStatPeakClus(slmRh, BinMaskRh, Thre);

NclusLh = length(ClusLh.clusid); NclusRh = length(ClusRh.clusid);   % total number of clusters

% check if permutation inference is needed
if NclusLh == 0 && NclusRh == 0
    disp('----- No Cluter Above the Threshold -----')
    ClusPLh = 'NA'; ClusPRh = 'NA';
    return
elseif Nperm == 0
    ClusPLh = 'NA'; ClusPRh = 'NA';
    return
end

disp('----- MEGHA Cluster Inference -----')
TSurfLh = U'*SurfLh; TSurfRh = U'*SurfRh;   % transformed phenotypes
TP0 = eye(TNsubj); TNcov = 0;   % transformed projection matrix and number of covariates

TErrLh = TP0*TSurfLh; TErrRh = TP0*TSurfRh;   % compute the transformed residuals
TSigma0Lh = sum(TErrLh.^2)'/(TNsubj-TNcov); TSigma0Rh = sum(TErrRh.^2)'/(TNsubj-TNcov);   % estimate of the transformed residual variance

clear SurfLh; clear SurfRh; clear TSurfLh; clear TSurfRh;   % save RAM

% permutation
slmPermLh.t = zeros(1,NvetTotLh); slmPermRh.t = zeros(1,NvetTotRh);   % allocate space
slmPermLh.tri = TriLh.tri; slmPermRh.tri = TriRh.tri;

PermStatLh = zeros(1,NvetLh); PermStatRh = zeros(1,NvetRh);   % allocate space
MaxClusPermLh = zeros(1,Nperm); MaxClusPermRh = zeros(1,Nperm);   % allocate space

Nreps = 50; Nstep = ceil(Nperm/Nreps);
fprintf(['|', repmat('-',1,Nreps), '|\r '])   % make the background output

for s = 1:Nperm
    if rem(s,Nstep) == 0 || s == Nperm
        fprintf('*')   % morniter progress
    end
    % permute the transformed GRM
    if s == 1
        TKperm = TK;   % permutation samples should always include the observation
    else
        TPermSubj = randperm(TNsubj);
        TKperm = TK(TPermSubj,TPermSubj);
    end
    
    % compute permuted score test statistics
    for i = 1:NblkLh
        if i ~= NblkLh
            PermStatLh((i-1)*Nunit+1:i*Nunit) = diag(TErrLh(:,(i-1)*Nunit+1:i*Nunit)'*TKperm*TErrLh(:,(i-1)*Nunit+1:i*Nunit))./(2*TSigma0Lh((i-1)*Nunit+1:i*Nunit));
        else
            PermStatLh((i-1)*Nunit+1:end) = diag(TErrLh(:,(i-1)*Nunit+1:end)'*TKperm*TErrLh(:,(i-1)*Nunit+1:end))./(2*TSigma0Lh((i-1)*Nunit+1:end));
        end
    end
    
    for i = 1:NblkRh
        if i ~= NblkRh
            PermStatRh((i-1)*Nunit+1:i*Nunit) = diag(TErrRh(:,(i-1)*Nunit+1:i*Nunit)'*TKperm*TErrRh(:,(i-1)*Nunit+1:i*Nunit))./(2*TSigma0Rh((i-1)*Nunit+1:i*Nunit));
        else
            PermStatRh((i-1)*Nunit+1:end) = diag(TErrRh(:,(i-1)*Nunit+1:end)'*TKperm*TErrRh(:,(i-1)*Nunit+1:end))./(2*TSigma0Rh((i-1)*Nunit+1:end));
        end
    end
    
    % permuted cluster size
    slmPermLh.t(MaskLh) = PermStatLh; slmPermRh.t(MaskRh) = PermStatRh;
    
    [~,ClusPermLh,~] = SurfStatPeakClus(slmPermLh, BinMaskLh, Thre);
    [~,ClusPermRh,~] = SurfStatPeakClus(slmPermRh, BinMaskRh, Thre);
    
    if ~isempty(ClusPermLh)
        MaxClusPermLh(s) = max(ClusPermLh.nverts);
    end
    if ~isempty(ClusPermRh)
        MaxClusPermRh(s) = max(ClusPermRh.nverts);
    end
end
fprintf('\n')   % new line in the command window

% compute family-wise error corrected cluster inference p-values
MaxClusPerm = max([MaxClusPermLh; MaxClusPermRh]);
ClusPLh = zeros(1,NclusLh); ClusPRh = zeros(1,NclusRh);

for i = 1:NclusLh
    ClusPLh(i) = sum(MaxClusPerm>=ClusLh.nverts(i))/Nperm;
end
for i = 1:NclusRh
    ClusPRh(i) = sum(MaxClusPerm>=ClusRh.nverts(i))/Nperm;
end
%