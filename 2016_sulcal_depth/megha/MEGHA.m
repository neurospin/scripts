function [Pval, h2, SE, PermPval, PermFWEcPval, Nsubj, Npheno, Ncov] = MEGHA(PhenoFile, header, CovFile, delimiter, GRMFile, GRMid, Nperm, WriteStat, OutDir)
% This function implements massively expedited genome-wide heritability analysis (MEGHA).
%
% Input arguments -
% PhenoFile: a plain text file containing phenotypic data
% Col 1: family ID; Col 2: subject ID; From Col 3: numerical values
% header: 1 - PhenoFile contains a headerline
% 0 - PhenoFile does not contain a headerline
% CovFile: a plain text file containing covariates (intercept NOT included)
% If no covariate needs to be included in the model, set CovFile = ''
% Col 1: family ID; Col 2: subject ID; From Col 3: numerical values
% delimiter: delimiter used in the phenotypic file and covariates file
% GRMFile: a plain text file containing the lower trianglar elements of the GRM
% Col 1 & 2: indices of pairs of individuals;
% Col 3: number of non-missing SNPs; Col 4: the estimate of genetic relatedness 
% GRMid: a plain text file of subject IDs corresponding to GRMFile
% Col 1: family ID; Col 2: subject ID
%
% [Note: Subjects in PhenoFile, CovFile and GRMFile do not need to be exactly the same.
% This function will find the subjects in common in these files and 
% sort the order of the subjects.]
%
% Nperm: number of permutations; set Nperm = 0 if permutation inference is not needed; default Nperm = 0
% WriteStat: 1 - write point estimates and significance measures of heritability to the output directory "OutDir"
% 0 - do not write statistics; default WriteStat = 0; default output directory is the working directory
%
% Output arguments -
% Pval: MEGHA p-values
% h2: MEGHA estimates of heritability magnitude
% SE: estimated standard error of heritability magnitude
% PermPval: MEGHA permutation p-values
% PermFWEcPval: MEGHA family-wise error corrected permutation p-values
% Nsubj: total number of subjects in common
% Npheno: number of phenotypes
% Ncov: number of covariates (including intercept)
%
% Output files (if WriteStat=1) -
% MEGHAstat.txt: point estimates and significance measures of heritability
% for each phenotype written to the output directory "OutDir"

% check inputs
if nargin < 6
    error('Not enough input arguments')
elseif nargin == 6
    Nperm = 0; WriteStat = 0; OutDir = 'NA';
elseif nargin == 7
    WriteStat = 0; OutDir = 'NA';
elseif nargin == 8
    OutDir = 'NA';
elseif nargin > 9
    error('Too many input arguments')
end

disp('----- Parse Data Files -----')
[Pheno, Cov, K, Nsubj, Npheno, Ncov, PhenoNames] = ReadFile(PhenoFile, header, CovFile, delimiter, GRMFile, GRMid);   % read data files

disp(['----- ', num2str(Nsubj), ' Subjects, ', num2str(Npheno), ' Phenotypes, ', num2str(Ncov), ' Covariates -----'])

disp('----- Compute MEGHA p-values and Heritability Estimates -----')
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
else
    disp('----- MEGHA Permutation Inference -----')
    TPheno = U'*Pheno;   % transformed phenotypes
    TP0 = eye(TNsubj); TNcov = 0;   % transformed projection matrix and number of covariates

    TErr = TP0*TPheno;   % compute the transformed residuals
    TSigma0 = sum(TErr.^2)'/(TNsubj-TNcov);   % estimate of the transformed residual variance

    PermStat = zeros(Npheno, Nperm);   % allocate space
    
    Nreps = 50; Nstep = ceil(Nperm/Nreps);
    fprintf(['|', repmat('-',1,Nreps), '|\r '])   % make the background output
    
    seed = RandStream('mt19937ar','Seed',0);
    for s = 1:Nperm
        if rem(s,Nstep) == 0 || s == Nperm
            fprintf('*')   % morniter progress
        end
        % permute the transformed GRM
        if s == 1
            TKperm = TK;   % permutation samples should always include the observation
        else
            TPermSubj = randperm(seed,TNsubj);   % shuffle subject IDs
            TKperm = TK(TPermSubj,TPermSubj);   % permutate transformed GRM
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

    % compute MEGHA permutation p-values
    PermPval = ones(Npheno,1);   % allocate space
    for i = 1:Npheno
        PermPval(i) = sum(PermStat(i,:)>=PermStat(i,1))/Nperm;
    end

    % compute MEGHA family-wise error corrected permutation p-values
    PermFWEcPval = ones(Npheno,1);   % allocate space
    MaxPermStat = max(PermStat);   % compute maximum statistics for each permutation
    for i = 1:Npheno
        PermFWEcPval(i) = sum(MaxPermStat>=PermStat(i,1))/Nperm;
    end
end
fprintf('\n')   % new line in the command window

% write MEGHA statistics
if WriteStat == 1
    disp('----- Write MEGHA Statistics -----')
    WriteStats(OutDir, Npheno, Nperm, header, PhenoNames, Pval, h2, PermPval, PermFWEcPval);
end
%