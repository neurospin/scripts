function run_Compute_Quantif(Rawdatdir,subject,Processeddir,subjectnumber,Codedir,Pythonexe)
% run_Compute_Quantif('C:\Users\js247994\Documents\Bipli2\Biplisubjects','C:\Users\js247994\Documents\ReconstructionTPI','C:\Program Files (x86)\Python36-32\python.exe')


%try to get T1 values from excel file
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


if ~exist('Pythonexe','var')
    Pythonexe='python';
    if ~exist('Codedir','var')
        Codedir=fullfile(char(pwd),'ReconstructionTPI');
    end
end

%Check if the subject number of the patient is available, if not try to
%count the order of patients to get the right number
if ~exist('subjectnumber','var')
    S = dir(Rawdatdir);
    S=S(~ismember({S.name},{'.','..'}));
    %nbsubjects=sum([S.isdir]);
    i=0;
    for subj=S'
        if subj.isdir==1
            i=i+1;
                if subject==subj.name
                    if ( i > 9 ) 
                    subjectnumber =  int2str(i) ;
                    else
                        subjectnumber =  [ '0', int2str(i) ] ;
                    end
                end
        end
    end
end

if ~exist(Processeddir,'dir')
    mkdir(char(Processeddir))
    mkdir(fullfile(char(Processeddir,subject)))
else
    if ~exist(fullfile(char(Processeddir),char(subject)),'dir')
        mkdir(fullfile(char(Processeddir),char(subject)));
    end
end

Subjectdirr=fullfile(char(Rawdatdir),char(subject));
Subjectdirtwix=fullfile(Subjectdirr,'twix7T');
Subjectdirp=fullfile(char(Processeddir),char(subject));
Subjectdirresult=fullfile(Subjectdirp,'TPI','Reconstruct_gridding','01-Raw');

%Create the processeddir subject folders if they are not already created
if ~exist(fullfile(Subjectdirp,'Anatomy3T'),'dir')
    mkdir(fullfile(Subjectdirp,'Anatomy3T'));
end
if ~exist(fullfile(Subjectdirp,'Anatomy7T'),'dir')
    mkdir(fullfile(Subjectdirp,'Anatomy7T'));
end
if ~exist(fullfile(Subjectdirp,'Anatomy3T'),'dir')
    mkdir(fullfile(Subjectdirp,'Anatomy3T'));
end
if ~exist(fullfile(Subjectdirp,'TPI'),'dir')
    mkdir(fullfile(Subjectdirp,'TPI'));
    mkdir(fullfile(Subjectdirp,'TPI','Reconstruct_gridding','01-Raw'));
    mkdir(fullfile(Subjectdirp,'TPI','Reconstruct_gridding','02-Post-Quantif'));
    mkdir(fullfile(Subjectdirp,'TPI','Reconstruct_gridding','03-Filtered'));
    mkdir(fullfile(Subjectdirp,'TPI','Reconstruct_gridding','04-7Tanatspace'));
    mkdir(fullfile(Subjectdirp,'TPI','Reconstruct_gridding','05-3Tanatspace'));
    mkdir(fullfile(Subjectdirp,'TPI','Reconstruct_gridding','06-MNIspace'));
end
if ~exist(fullfile(Subjectdirp,'Trufi'),'dir')
    mkdir(char(fullfile(Subjectdirp,'Trufi')));
end

%Launch the reconstruction of the dat files found in raw to the processed
%images and placed them in right location (should perhaps eventually be
%updated to also include FISTA?
reconstructfile=fullfile(char(Codedir),'ProcessData.py');
launch_reconstruct_TPI(Subjectdirtwix,Subjectdirresult,subjectnumber,reconstructfile,Pythonexe );


filesdirin=fullfile(Subjectdirp,'TPI','Reconstruct_gridding','01-Raw');
filesdirout=fullfile(Subjectdirp,'TPI','Reconstruct_gridding','02-PostQuantif');
filesdone=[];
filesnii=dir(fullfile(filesdirin,'*.nii'));

ComputeVFAfile=fullfile(char(Codedir),'ComputeDensity3D_clinic.py');
ComputeQuantifile=fullfile(char(Codedir),'ComputeQuantif_clinic.py');

for l=subjstartl:size(raw,1)
    if strcmp(raw(l,subjcol),subject)
        T1valtest=raw(l,T1col);
        T1valtest=T1valtest{1};
    end
end

T1val=3.947000;

for file1=filesnii'
    degloc=strfind(file1.name,'deg');
    over=0;
    for filesdon=filesdone'
        if strcmp(file1.name,filesdon')
            over=1;
        end
    end
    if ~isempty(degloc)
        file1path=fullfile(filesdirin,file1.name);
        deg1=file1.name(degloc-2:degloc-1);   
        if ~over   
            for file2=filesnii'
                VFA=~strcmp(file1.name,file2.name) && strcmp(file1.name(degloc:end),file2.name(degloc:end)); %If there is another file with the same name but a different degree value, it is treated as the second VFA file
                if VFA
                    deg2=file2.name(degloc-2:degloc-1);
                    Computedniipath=fullfile(filesdirout,strcat(file1.name(1:degloc-3),file1.name(degloc+4:end)));
                    file2path=fullfile(filesdirin,file2.name);
                    codelaunchVFA=strcat({'"'},Pythonexe,{'" '},ComputeVFAfile,{' --i1 '},file1path,{' --i2 '},file2path,{' --deg1 '},deg1,{' --deg2 '},deg2,{' --t1 '}, num2str(T1val), {' --v --o '},Computedniipath);
                    status = system(codelaunchVFA{1});
                    filesdone=[filesdone;file1.name;file2.name];
                end
            end
        end
        %[~,filename,ext]=fileparts(file1.name);
        %Computedniipath=strcat(filesdir,filename,ext);
        Computedniipath=fullfile(filesdirout,file1.name);
        codelaunchQuant=strcat({'"'},Pythonexe,{'" '},ComputeQuantifile,{' --i '},file1path,{' --deg '},deg1,{' --t1 '}, num2str(T1val), {' --v --o '},Computedniipath);
        status = system(codelaunchQuant{1});
    end
end

  