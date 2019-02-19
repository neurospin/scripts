function run_create_processfolders(Processeddir,subject,reconstruct_type)

    Processeddir=string(Processeddir);
    subject=string(subject);
    if ~exist(Processeddir,'dir')
        mkdir(string(Processeddir))
        mkdir(fullfile(Processeddir,subject))
    else
        if ~exist(fullfile(char(Processeddir),char(subject)),'dir')
            mkdir(fullfile(char(Processeddir),char(subject)));
        end
    end
    if ~exist('reconstruct_type','var')
        reconstruct_type='Reconstruct_gridding';
    end

 %   Subjectdirr=fullfile(char(Rawdatdir),char(subject));
   % Subjectdirtwix=fullfile(Subjectdirr,'twix7T');
    Subjectdirp=fullfile(char(Processeddir),char(subject));
 %   Subjectdirresult=fullfile(Subjectdirp,'TPI','Reconstruct_gridding','01-Raw');

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
    if ~exist(fullfile(Subjectdirp,'TPI',reconstruct_type,'01-Raw'),'dir')
        mkdir(fullfile(Subjectdirp,'TPI',reconstruct_type,'01-Raw'));
    end
    if ~exist(fullfile(Subjectdirp,'TPI',reconstruct_type,'05-MNIspace'),'dir')
        mkdir(fullfile(Subjectdirp,'TPI',reconstruct_type,'02-PostQuantif'));
        mkdir(fullfile(Subjectdirp,'TPI',reconstruct_type,'03-Filtered'));
        mkdir(fullfile(Subjectdirp,'TPI',reconstruct_type,'04-3Tanatspace'));
        mkdir(fullfile(Subjectdirp,'TPI',reconstruct_type,'05-MNIspace'));
    end
end