function WriteMap(SurfDir, FSDir, SubjID, ImgFileLh, ImgFileRh, OutDir, PvalLh, PvalRh, h2Lh, h2Rh)
% This function writes surface maps of MEGHA heritability estimates and
% -log10(p-values) to the output directory.
%
% Input arguments -
% SurfDir: directory of the surface data (SUBJECTS_DIR)
% FSDir: directory of FreeSurfer where the folder "subjects" can be found (FREESURFER_DIR)
% SubjID: an Nsubj x 1 cell structure containing a list of subject IDs
% ImgFileLh/ImgFileRh: name of the file containing surface data for the left/right hemisphere
% OutDir: output directory; default directory is the working directory
% PvalLh/PvalRh: MEGHA p-values for in-mask vertices on the left/right hemisphere
% h2Lh/h2Rh: MEGHA heritability estimates for in-mask vertices on the left/right hemisphere
%
% Output files -
% LogPvalLh.mgh/LogPvalRh.mgh and h2Lh.mgh/h2Rh.mgh written to the output directory "OutDir"

logPvalLh = -log10(PvalLh); logPvalRh = -log10(PvalRh);   % compute -log10(p-values)
    
% read mask files
fid = fopen([FSDir, 'subjects/fsaverage/label/lh.cortex.label']);   % open mask file for the lh
LabelLh = textscan(fid,'%u %f %f %f %f','Headerlines',2);   % skip the first and second line and read the rest of the file
MaskLh = LabelLh{1}+1;   % convert 0-based index to 1-based

fid = fopen([FSDir, 'subjects/fsaverage/label/rh.cortex.label']);   % open mask file for the rh
LabelRh = textscan(fid,'%u %f %f %f %f','Headerlines',2);   % skip the first and second line and read the rest of the file
MaskRh = LabelRh{1}+1;   % convert 0-based index to 1-based
    
% write image
TempImg = 1;   % use the first subject's image as a template

FileLh = [SurfDir, SubjID{TempImg}, '/surf/', ImgFileLh];   % file name
DataLh = MRIread(FileLh);   % read file
DataLh.vol(:) = -1;   % remove all values
    
DataLh.vol(MaskLh) = logPvalLh;   % fill in lh -log Pval
if strcmp(OutDir, 'NA')
    MRIwrite(DataLh, 'LogPvalLh.mgh');   % write image
else
    MRIwrite(DataLh, [OutDir, 'LogPvalLh.mgh']);   % write image
end
    
DataLh.vol(MaskLh) = h2Lh;   % fill in lh h2 estimates
if strcmp(OutDir, 'NA')
    MRIwrite(DataLh, 'h2Lh.mgh');   % write image
else
    MRIwrite(DataLh, [OutDir, 'h2Lh.mgh']);   % write image
end
%
FileRh = [SurfDir, SubjID{TempImg}, '/surf/', ImgFileRh];   % file name
DataRh = MRIread(FileRh);   % read file
DataRh.vol(:) = -1;   % remove all values
    
DataRh.vol(MaskRh) = logPvalRh;   % fill in rh -log Pval
if strcmp(OutDir, 'NA')
    MRIwrite(DataRh, 'LogPvalRh.mgh');   % write image
else
    MRIwrite(DataRh, [OutDir, 'LogPvalRh.mgh']);   % write image
end
    
DataRh.vol(MaskRh) = h2Rh;   % fill in rh h2 estimates
if strcmp(OutDir, 'NA')
    MRIwrite(DataRh, 'h2Rh.mgh');   % write image
else
    MRIwrite(DataRh, [OutDir, 'h2Rh.mgh']);   % write image
end
%