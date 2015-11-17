function [NvetLh, NvetRh, SurfLh, SurfRh] = ExtractSurf(SurfDir, SubjID, ImgFileLh, ImgFileRh, FSDir)
% This function extracts surface data resampled on fsaverage 
% organized in the convention of FreeSurfer.
%
% Input arguments -
% SurfDir: directory of the surface data (SUBJECTS_DIR)
% SubjID: a plain text file containing a list of subject IDs to be included in the analysis
% ImgFileLh/ImgFileRh: name of the file containing surface data for the left/right hemisphere
% FSDir: directory of FreeSurfer where the folder "subjects" can be found (FREESURFER_DIR)
%
% Output arguments -
% NvetLh/NvetRh: number of in-mask vertices on the left/right hemisphere
% SurfLh/SurfRh: an Nsubj x NvetLh/NvetRh data matrix for the left/right hemisphere

% read mask files
fid = fopen([FSDir, 'subjects/fsaverage/label/lh.cortex.label']);   % open mask file for the lh
LabelLh = textscan(fid,'%u %f %f %f %f','Headerlines',2);   % skip the first and second line and read the rest of the file
MaskLh = LabelLh{1}+1;   % convert 0-based index to 1-based
NvetLh = length(MaskLh);   % number of in-mask vertices for the left hemisphere

fid = fopen([FSDir, 'subjects/fsaverage/label/rh.cortex.label']);   % open mask file for the rh
LabelRh = textscan(fid,'%u %f %f %f %f','Headerlines',2);   % skip the first and second line and read the rest of the file
MaskRh = LabelRh{1}+1;   % convert 0-based index to 1-based
NvetRh = length(MaskRh);   % number of in-mask vertices for the right hemisphere

% read surface data
Nsubj = length(SubjID);   % number of subjects
SurfLh = zeros(Nsubj,NvetLh); SurfRh = zeros(Nsubj,NvetRh);   % allocate space

for i = 1:Nsubj
    FileLh = [SurfDir, SubjID{i}, '/surf/', ImgFileLh];   % file name
    StructLh = MRIread(FileLh);   % read file
    StructLh.vol(isnan(StructLh.vol)) = 0;   % set all NaN to zero
    SurfLh(i,:) = StructLh.vol(MaskLh);   % apply the mask
    
    FileRh = [SurfDir, SubjID{i}, '/surf/', ImgFileRh];   % file name
    StructRh = MRIread(FileRh);   % read file
    StructRh.vol(isnan(StructRh.vol)) = 0;   % set all NaN to zero
    SurfRh(i,:) = StructRh.vol(MaskRh);   % apply the mask
end
%