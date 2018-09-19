function launch_reconstruct_TPI_all(projectdir)%,subjectdir,subjectnumber,reconstructfile,Pythonexe )

    %launch_reconstruct_TPI('C:\Users\js247994\Documents\Bipli2\Test8\Raw','C:\Users\js247994\Documents\Bipli2\Test8\Processed','C:\Users\js247994\Documents\Bipli2\BipliPipeline\scripts\2017_bipli_lithium_imaging\ReconstructionTPI','C:\Python27\python.exe')
    if ~exist('Pythonexe','var')
        pythonexe='python';
        if ~exist('Codedir','var')
            codedir=fullfile(char(pwd));
        end
    end
    
    projectdir='V:\projects\BIPLi7\ClinicalData';
    projectdir='/neurospin/ciclops/projects/BIPLi7/ClinicalData';
    raw_dir=fullfile(projectdir,'Raw_Data','2017_05_16*');
%    reconstructfile=fullfile('/home/js247994/DocumentsN2/2017_bipli_lithium_imaging','ReconstructionTPI','ProcessData.py');
    listsubj=dir(raw_dir);

    excelT1s=fullfile(pwd,'info_pipeline','T1vals.xlsx');
    if exist(excelT1s,'file')
        [~,~,raw]=xlsread(excelT1s);
        for l=1:size(raw,1)
            for m=1:size(raw,2)
                if strcmp('subj',raw(l,m))
                    subjcol=m;
                    subjstartl=l+1;
                elseif strcmp('T1 vals',raw(l,m))
                    T1col=m;
                end
            end
        end
    end      
    %try to get T1 values from excel file  
    
    for i = 1:numel(listsubj)
        subjname=listsubj(i).name;    
        
        for l=subjstartl:size(raw,1)
            if strcmp(raw(l,subjcol),subjname)
                T1val=raw(l,T1col);
                T1val=T1val{1};
            end
        end
        
        if i<10
            subjectnumber=strcat('0',int2str(i));
        elseif i>=10
            subjectnumber=int2str(i);
        end       
        %maybe one day will be changed to actually include the value from
        %the file?
        T1val=3.947000;
        launch_reconstruct_TPI(projectdir,subjname,codedir,T1val,pythonexe,subjectnumber)
        %Check if the subject number of the patient is available, if not try to
        %count the order of patients to get the right number

    end
end
