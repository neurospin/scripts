pathFolder = 'C:\Users\js247994\Downloads\2018_05_16\DICOM\H_highfft\';
searchFolder=strcat(pathFolder,'*Hz*');
resultFolder=strcat(pathFolder,'Allniis');
d = dir(searchFolder);
isub = [d(:).isdir ]; %# returns logical vector
nameFolds = {d(isub).name}';
for i= 1:length(nameFolds)
    name=nameFolds{i};
    dicm2nii(strcat(pathFolder,name),resultFolder,'.nii')
end
dicm2nii('C:\Users\js247994\Downloads\2018_05_16\DICOM\H_highfft\136Hz_H',resultFolder,'.nii');
movefile( fullfile(projectdir, oldnames{K}), fullfile(projectdir, newnames{K}) );