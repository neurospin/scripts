function run_Compute_Quantif(Processeddir,Codedir,Pythonexe)

% run_Compute_Quantif('C:\Users\js247994\Documents\Bipli2\Biplisubjects','C:\Users\js247994\Documents\ReconstructionTPI','C:\Program Files (x86)\Python36-32\python.exe')

if ~exist('Pythonexe','var')
    Pythonexe='python';
    if ~exist('Codedir','var')
        Codedir=strcat(pwd,'\ReconstructionTPI');
    end
end

S = dir(Processeddir);
S=S(~ismember({S.name},{'.','..'}));
%nbsubjects=sum([S.isdir]);
i=0;

excelT1s=strcat(pwd,'\info_pipeline\T1vals.xlsx');
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

for subj=S'
    if subj.isdir==1
        i=i+1;
        subject=subj.name;
        ComputeVFAfile=strcat(Codedir,'\ComputeDensity3D_clinic.py');
        ComputeQuantifile=strcat(Codedir,'\ComputeQuantif_clinic.py');
        filesdirin=strcat(Processeddir,'\',subject,'\TPI\Reconstruct_gridding\01-Raw\');
        filesdirout=strcat(Processeddir,'\',subject,'\TPI\Reconstruct_gridding\02-PostQuantif\');
        filesdone=[];
        filesnii=dir(strcat(filesdirin,'*.nii'));
        
        for l=subjstartl:size(raw,1)
            if strcmp(raw(l,subjcol),subject)
                T1valtest=raw(l,T1col);
                T1valtest=T1valtest{1};
            end
        end
        
        T1val=4.56;
        for file1=filesnii'
            degloc=strfind(file1.name,'deg');
            over=0;
            for filesdon=filesdone'
                if strcmp(file1.name,filesdon')
                    over=1;
                end
            end
            if ~isempty(degloc)
                file1path=strcat(filesdirin,file1.name);
                deg1=file1.name(degloc-2:degloc-1);   
                if ~over   
                    for file2=filesnii'
                        VFA=~strcmp(file1.name,file2.name) && strcmp(file1.name(degloc:end),file2.name(degloc:end)); %If there is another file with the same name but a different degree value, it is treated as the second VFA file
                        if VFA
                            deg2=file2.name(degloc-2:degloc-1);
                            Computedniipath=strcat(filesdirout,file1.name(1:degloc-3),file1.name(degloc+4:end));
                            file2path=strcat(filesdirin,file2.name);
                            codelaunchVFA=strcat({'"'},Pythonexe,{'" '},ComputeVFAfile,{' --i1 '},file1path,{' --i2 '},file2path,{' --deg1 '},deg1,{' --deg2 '},deg2,{' --t1 '}, num2str(T1val), {' --v --o '},Computedniipath);
                            status = system(codelaunchVFA{1});
                            filesdone=[filesdone;file1.name;file2.name];
                        end
                    end
                end
                %[~,filename,ext]=fileparts(file1.name);
                %Computedniipath=strcat(filesdir,filename,ext);
                Computedniipath=strcat(filesdirout,file1.name);
                codelaunchQuant=strcat({'"'},Pythonexe,{'" '},ComputeQuantifile,{' --i '},file1path,{' --deg '},deg1,{' --t1 '}, num2str(T1val), {' --v --o '},Computedniipath);
                status = system(codelaunchQuant{1});
            end
        end
        
    end    
end
  