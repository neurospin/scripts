function [Nsubj, Ncol, SubjID, Data, PhenoNames] = ParseFile(filename, delimiter, header)
% This function parses a plain text file with specified delimiter with no missing values.
%
% Input arguments - 
% filename: directory and name of the text file
% Col 1: family ID; Col 2: subject ID; From Col 3: numerical values
% delimiter: delimiter used in the text file
% header: 1 - PhenoFile contains a headerline
% 0 - PhenoFile does not contain a headerline; default header = 0
%
% Output arguments -
% Nsubj: total number of subjects
% Ncol: number of quantitative variables
% SubjID: a list of subject IDs
% Data: an Nsubj x Ncol matrix for quantitative data
% PhenoNames: names of the phenotypes in the headerline

% check inputs
if nargin < 3
    header = 0;
end
 
delimiter = sprintf(delimiter);   % delimiter

fid = fopen(filename);   % open the file

SubjID = textscan(fid, '%*s %s %*[^\n]', 'Delimiter', delimiter, 'HeaderLines', header);   % read the second column (subject ID)
SubjID = SubjID{1};   % convert a 1x1 cell structure into a Nsubj x 1 cell structure
Nsubj = length(SubjID);   % calculate number of subjects

frewind(fid)   % set the file position indicator to the beginning of the file
line = fgetl(fid);   % read the first line of the data
Ncol = numel(strfind(line, delimiter)) - 1;   % calculate the number of columns for quantitative data

frewind(fid)   % set the file position indicator to the beginning of the file
Data = cell2mat(textscan(fid, ['%*s %*s', repmat('%f',1,Ncol)], 'Delimiter', delimiter, 'HeaderLines', header));   % read quantitative data

frewind(fid)   % set the file position indicator to the beginning of the file
if header == 1    
    PhenoNames = textscan(fid, ['%*s %*s', repmat('%s',1,Ncol)], 1, 'Delimiter', delimiter);   % get the names of the phenotypes
else
    PhenoNames = 'NA';   % no headerline
end

fclose(fid);   % close the file
%