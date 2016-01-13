function [Nsubj, SubjID, K] = ParseGRM(GRMFile, GRMid)
% This function constructs the genetic relationship matrix (GRM) 
% from a plain text file containing the lower trianglar elements of the GRM
% and a plain text file containing subject IDs.
%
% Input arguments - 
% GRMFile: a plain text file containing the lower triangle elements of the GRM with no headline
% Col 1 & 2: indices of pairs of individuals; 
% Col 3: number of non-missing SNPs; Col 4: the estimate of genetic relatedness 
% GRMid: a plain text file of subject IDs with no headline corresponding to GRMFile
% Col 1: family ID; Col 2: subject ID
%
% Output arguments -
% Nsubj: total number of subjects
% SubjID: a list of subject IDs
% K: an Nsubj x Nsubj GRM
                                                          
fid = fopen(GRMFile);   % open the GRM file
GRM = textscan(fid,'%f %f %f %f');   % read the GRM file
fclose(fid);   % close the GRM file

Row = GRM{1}; Col = GRM{2}; Value = GRM{4};   % parse the GRM file
Nentry = length(Value);   % number of lower triangle elements of the GRM matrix
Nsubj =1/2*(-1+sqrt(1+8*Nentry));   % calculate the number of subjects

K = zeros(Nsubj,Nsubj);   % allocate space
for i = 1:Nentry   % construct GRM
    K(Row(i),Col(i)) = Value(i);
    K(Col(i),Row(i)) = Value(i);   % GRM is symmetric
end

fid = fopen(GRMid);   % open the file containing subject IDs 
SubjID = textscan(fid,'%*s %s');   % read the column for subject IDs
fclose(fid);   % close the GRM ID file
SubjID = SubjID{1};   % convert a 1x1 cell structure into a Nsubj x 1 cell structure
%