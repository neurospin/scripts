pathFolder = 'V:\projects\BIPLi7\Tests\2018_05_29\';

searchFolder=strcat(pathFolder,'\*DEG');
resultFolder=strcat(pathFolder,'Allniis');
d = dir(searchFolder);
isub = [d(:).isdir ]; %# returns logical vector
nameFolds = {d(isub).name}';
for i= 1:length(nameFolds)
    name=nameFolds{i};
    dicm2nii(strcat(pathFolder,name),resultFolder,'.nii')
    %movefile( fullfile(resultFolder,'trufi_RR.nii'), fullfile(resultFolder, strcat(name, '.nii') ) );
    movefile( fullfile(resultFolder,'dcmHeaders.mat'), fullfile(resultFolder, strcat(name, '.mat') ));
end
%dicm2nii('C:\Users\js247994\Downloads\2018_05_16\DICOM\H_highfft\136Hz_H',resultFolder,'.nii');
%dicm2nii('V:\projects\BIPLi7\Tests\2018_05_23\bw136_10deg',