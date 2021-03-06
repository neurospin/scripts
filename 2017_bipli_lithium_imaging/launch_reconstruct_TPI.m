 function launch_reconstruct_TPI(projectdir,subjname,codedir,T1val,pythonexe,subjectnumber,reconstruct_type)
   
    if exist(fullfile(projectdir,"Raw_Data",subjname,"twix7T"),'dir')
        raw_subjdir=fullfile(projectdir,"Raw_Data",subjname,"twix7T");
    elseif exist(fullfile(projectdir,"Raw_Data",subjname,"twix"),'dir')
        raw_subjdir=fullfile(projectdir,"Raw_Data",subjname,"twix");
    end
    proc_subjdir=fullfile(projectdir,"Processed_Data",subjname);
    reconstructfile=fullfile(codedir,"reconstructionTPI","ProcessData.py");
    
    if ~exist('subjectnumber','var')
        S = dir(raw_subjdir);
        S=S(~ismember({S.name},{'.','..'}));
        %nbsubjects=sum([S.isdir]);
        i=0;
        for subj=S'
            if subj.isdir==1
                i=i+1;
                if subject==subjname
                    if ( i > 9 ) 
                        subjectnumber =  int2str(i) ;
                    else
                        subjectnumber =  [ "0", int2str(i) ] ;
                    end
                end
            end
        end
    end  
    
    %getanatfiles(fullfile(projectdir,"Raw_Data",subjname),proc_subjdir);
    voxres=32;
    
    run_create_processfolders(fullfile(projectdir,'Processed_Data'),subjname,reconstruct_type)
    listdat=dir(raw_subjdir);
    
    launch_reconstruct=0;
    launch_post_process=1;
    
    if launch_reconstruct
        for j=1:numel(listdat)
            ok=strfind(listdat(j).name,'TPI');
            if ~isempty(ok)
                processedTPIpath=fullfile(proc_subjdir,"TPI",strcat(reconstruct_type,""),"01-Raw");
                if ~exist(processedTPIpath,'dir')
                    mkdir(processedTPIpath)
                end
                Tpifilename=listdat(j).name;
                Tpifilepath=fullfile(raw_subjdir,char(Tpifilename));
                deg=strfind(Tpifilename,'deg');
                degval=(Tpifilename(deg-2:deg-1));
                MID=strfind(Tpifilename,'MID');
                %MIDval=Tpifilename(MID+3:MID+6);%*isstrprop(Tpifilename(MID+3:MID+5),'alphanum');
                MIDval="";
                for l=MID+3:MID+6
                    if isstrprop(Tpifilename(l),'alphanum')
                        MIDval=MIDval+Tpifilename(l);
                    end
                end
                    
                TPIresultname=strcat("Patient",(subjectnumber),"_",degval,"deg_MID",MIDval);
                Reconstructpath=fullfile(processedTPIpath,(TPIresultname+".nii"));
                if contains(reconstruct_type,'Reconstruct_gridding')
                    codelaunch=pythonexe+" "+reconstructfile+" --i "+Tpifilepath+" --NSTPI --s --o "+Reconstructpath;
                    %system(codelaunch);
                elseif contains(reconstruct_type,'Reconstruct_sandro')
                    reconTPI(voxres, 'none', Tpifilepath, Reconstructpath);
                            %reconTPI(voxres, 'none', Tpifilepath, processedTPIpath);
                else
                    disp('reconstruct type unrecognized');
                    raise error
                end

                %system(codelaunch{1,1})

                fieldmap_file=fullfile(proc_subjdir,"Field_mapping","fieldmap_final.nii");
                spm_alignmentfile=fullfile(codedir,"info_pipeline","fieldmapwritespm.mat");
                %ref_im=dir(fullfile(projectdir,'*.nii'));
                %ref_im=fullfile(projectdir,string(ref_im.name));
                ref_dir=(fullfile(processedTPIpath,TPIresultname+"*.nii"));
                ref_im=dir(ref_dir);
                forcefieldmap=0;
                %system(codelaunch{1,1})

                fieldmap_file=fullfile(proc_subjdir,"Field_mapping","fieldmap_final.nii");
                spm_alignmentfile=fullfile(codedir,"info_pipeline","fieldmapwritespm.mat");
                %ref_im=dir(fullfile(projectdir,'*.nii'));
                %ref_im=fullfile(projectdir,string(ref_im.name));
                ref_dir=(fullfile(processedTPIpath,TPIresultname+"*.nii"));
                ref_im=dir(ref_dir);
                forcefieldmap=1;
                if ~exist(fieldmap_file,'file') || forcefieldmap
                    if ~isempty(ref_im)
                        ref_im=string(fullfile(processedTPIpath,ref_im(1).name));
                        %prepare_fieldmap(projectdir,subjname,spm_alignmentfile,ref_im);   
                    else
                        disp('warning, could not find reference image for fieldmap');
                        %disp('reconstruct type unrecognized');
                        %raise error
                    end
                end

                TPIresultfname=("Patient"+(subjectnumber)+"_"+degval+"deg_MID"+MIDval+"_B0cor.nii");
                Reconstructfpath=fullfile(processedTPIpath,TPIresultfname);
                if strcmp(reconstruct_type,'Reconstruct_gridding')
                    %codelaunch=pythonexe+" "+reconstructfile+" --i "+Tpifilepath+" --fieldmap "+fieldmap_file+" --NSTPI --s --FISTA_CSV --o "+Reconstructfpath;
                    codelaunch=pythonexe+" "+reconstructfile+" --i "+Tpifilepath+" --fieldmap "+fieldmap_file+" --NSTPI --s --o "+Reconstructfpath;
                    %system(codelaunch)
                    disp('hi');
                elseif contains(reconstruct_type,'Reconstruct_sandro')
                    filter=check_reconstruct_type(reconstruct_type);
                    reconTPI_B0cor(voxres,filter,Tpifilepath,fieldmap_file,Reconstructfpath,'fsc',128); %128               
                else
                    disp('reconstruct type unrecognized');
                    raise error
                end

                TPIresultfname=("Patient"+(subjectnumber)+"_"+degval+"deg_MID"+MIDval+"_B0cor.nii");
                Reconstructfpath=fullfile(processedTPIpath,TPIresultfname);
                if contains(reconstruct_type,'Reconstruct_gridding')
                    codelaunch=pythonexe+" "+reconstructfile+" --i "+Tpifilepath+" --fieldmap "+fieldmap_file+" --NSTPI --s --o "+Reconstructfpath;
                    %system(codelaunch)
                elseif contains(reconstruct_type,'Reconstruct_sandro')
                    reconTPI_B0cor(voxres,'none',Tpifilepath,fieldmap_file,Reconstructfpath,'fsc',16); %128               
                else
                    disp('reconstruct type unrecognized');
                    raise error
                    %system(codelaunch{1,1})
                end
            end
        end
    end
    %fullstudy=0;
    %if fullstudy
    %    raw_dic=fullfile(projectdir,'Raw_Data',subjname,'DICOM7T'); 
    %end
    
    if launch_post_process
        raw_dic=fullfile(projectdir,'Raw_Data',subjname,'DICOM7T');  

        trufiproc=0;
        quantifproc=1;
        BTKproc=0;
        force_regcalc=0; %Keep this at 0 unless you want it do recalculate the segmentation fo rno reason
        forcestart=1; %some/most of the code does not write files if it notices they already exist. Putting this to one forces it to run anyway
        %if workontpi and/or workontrufi are also turned on
        workontpi=1;
        workontrufi=0;

        if trufiproc==1
            trufitoprocess(raw_dic,proc_subjdir);
        end

        transfoparam_file=launch_transform_calc(proc_subjdir,quantifproc,force_regcalc,reconstruct_type);
        if quantifproc==1
            trufiexcel=fullfile(projectdir,'Analysis','TPI_trufi_compare.xlsx');
            run_Compute_Quantif_2(fullfile(projectdir,'Processed_Data'),subjname,T1val,reconstruct_type,workontpi,workontrufi) %trufiexcel
        end
        if BTKproc==1
            runBTK(proc_subjdir,reconstruct_type,workontpi,workontrufi);
        end
        launch_applytransforms(proc_subjdir,transfoparam_file,reconstruct_type,forcestart,workontpi,workontrufi)
        if quantifproc==1
            %run_Compute_Quantif_3(fullfile(projectdir,'Processed_Data'),subjname,T1val,reconstruct_type)
        end
    end



