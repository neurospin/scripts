function [Pval, h2, SE, PermPval, PermFWEcPval, Nsubj, Npheno, Ncov] = MEGHAmat(Pheno, Cov, K, Nperm)
% This function implements massively expedited genome-wide heritability analysis (MEGHA) 
% when phenotypic data, covariates and GRM have been prepared as Matlab .mat format.
%
% Input arguments -
% Pheno: an Nsubj x Npheno phenotypic data matrix
% Cov: an Nsubj x Ncov covariates matrix (intercept included)
% K: an Nsubj x Nsubj GRM
%
% [Note: Subjects in Pheno, Cov and K must be exactly the same and arranged in the same order.]
%
% Nperm: number of permutations; set Nperm = 0 if permutation inference is not needed; default Nperm = 0 
%
% Output arguments -
% Pval: MEGHA p-values
% h2: MAGHA estimates of heritability magnitude
% SE: estimated standard error of heritability magnitude
% PermPval: MEGHA permutation p-values
% PermFWEcPval: MEGHA family-wise error corrected permutation p-values
% Nsubj: total number of subjects
% Npheno: number of phenotypes
% Ncov: number of covariates (including intercept)

% check inputs
if nargin < 3
    error('Not enough input arguments')
elseif nargin == 3
    Nperm = 0;
elseif nargin > 4
    error('Too many input arguments')
end

disp('----- Parse Data Files -----')
[Nsubj, Npheno] = size(Pheno);   % calculate the number of subjects and the number of phenotypes
[~, Ncov] = size(Cov);   % calculate the number of covariates

disp(['----- ', num2str(Nsubj), ' Subjects, ', num2str(Npheno), ' Phenotypes, ', num2str(Ncov), ' Covariates -----'])

disp('----- Compute MEGHA p-values and heritability estimates -----')
P0 = eye(Nsubj) - Cov*((Cov'*Cov)\Cov');   % compute the projection matrix

% Satterthwaite approximation
delta = trace(P0*K)/2;
I11 = trace(P0*K*P0*K)/2; I12 = trace(P0*K*P0)/2; I22 = trace(P0*P0)/2;
I = I11-I12^2/I22;   % efficient information
kappa = I/(2*delta); nv = 2*delta^2/I;   % compute the scaling parameter and the degrees of freedom

Err = P0*Pheno;   % compute the residuals
Sigma0 = sum(Err.^2)'/(Nsubj-Ncov);   % estimate of residual variance

Nunit = 100; Nblk = ceil(Npheno/Nunit);  % the size of block and total number of blocks 
Stat = zeros(Npheno,1);   % allocate space

% compute score test statistics
for i = 1:Nblk
    if i ~= Nblk
        Stat((i-1)*Nunit+1:i*Nunit) = diag(Err(:,(i-1)*Nunit+1:i*Nunit)'*K*Err(:,(i-1)*Nunit+1:i*Nunit))./(2*Sigma0((i-1)*Nunit+1:i*Nunit));
    else
        Stat((i-1)*Nunit+1:end) = diag(Err(:,(i-1)*Nunit+1:end)'*K*Err(:,(i-1)*Nunit+1:end))./(2*Sigma0((i-1)*Nunit+1:end));
    end
end

Pval = 1-chi2cdf(Stat/kappa,nv);   % compute MEGHA p-values

[U,~,~] = svd(P0); U = U(:,1:Nsubj-Ncov);   % singular value decomposition
TK = U'*K*U;   % transformed GRM
TNsubj = Nsubj-Ncov;   % transformed number of subjects
stdK = std(TK(triu(true(TNsubj),1)));   % standard deviation of the off-diagonal elements of the transformed GRM

% compute estimates of heritability magnitude
WaldStat = chi2inv(max(1-2*Pval,0), 1);   % Wald statistic
% SE = 316/Nsubj;   % theoretical estimate of the standard error of heritability magnitude
SE = sqrt(2)/TNsubj/stdK;   % empirical estimate of the standard error of heritability magnitude
h2 = min(SE*sqrt(WaldStat), 1);   % heritability estimates

% check if permutation inference is needed
if Nperm == 0
    PermPval = 'NA'; PermFWEcPval = 'NA';
    return
end

disp('----- MEGHA Permutation Inference -----')
TPheno = U'*Pheno;   % transformed phenotypes
TP0 = eye(TNsubj); TNcov = 0;   % transformed projection matrix and number of covariates

TErr = TP0*TPheno;   % compute the transformed residuals
TSigma0 = sum(TErr.^2)'/(TNsubj-TNcov);   % estimate of the transformed residual variance

PermStat = zeros(Npheno, Nperm);   % allocate space

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
    for i = 1:Nblk
        if i ~= Nblk
            PermStat((i-1)*Nunit+1:i*Nunit,s) = diag(TErr(:,(i-1)*Nunit+1:i*Nunit)'*TKperm*TErr(:,(i-1)*Nunit+1:i*Nunit))./(2*TSigma0((i-1)*Nunit+1:i*Nunit));
        else
            PermStat((i-1)*Nunit+1:end,s) = diag(TErr(:,(i-1)*Nunit+1:end)'*TKperm*TErr(:,(i-1)*Nunit+1:end))./(2*TSigma0((i-1)*Nunit+1:end));
        end
    end
end
fprintf('\n')   % new line in the command window

% compute MEGHA permutation p-values
PermPval = ones(Npheno,1);
for i = 1:Npheno
    PermPval(i) = sum(PermStat(i,:)>=PermStat(i,1))/Nperm;
end

% compute MEGHA family-wise error corrected permutation p-values
PermFWEcPval = ones(Npheno,1); 
MaxPermStat = max(PermStat);   % compute maximum statistics for each permutation
for i = 1:Npheno
    PermFWEcPval(i) = sum(MaxPermStat>=PermStat(i,1))/Nperm;
end
%