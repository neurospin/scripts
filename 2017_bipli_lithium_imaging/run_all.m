Processeddir='V:\projects\BIPLi7\ClinicalData\Processed_Data\';
S=dir(Processeddir);
S=S(~ismember({S.name},{'.','..'}));

for subj=S'
    if subj.isdir==1
        subjpath=subj.name;
        launch_calculate_all(strcat(Processeddir,subjpath));
    end
end
        