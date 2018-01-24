
function transmat=calculate_translation_mat_01(Lifile)%,Litranslation)


    %anat7TH=spm_vol(Anatfile);
    %C = textscan(fopen(Litranslation),'%d');
    %HtoLi= double(C{1,1}(1:3));
    %offset=abs(anat7TH.mat(1:3,4));
    %tranlat=-(465-offset-HtoLi)+[-23;5;0];
    Lispm=spm_vol(Lifile);
    %Lispmoffset=(Lispm.dim-1).*[Lispm.mat(1,1),Lispm.mat(2,2),Lispm.mat(3,3)];
    %tranlat=-(Lispmoffset'-offset-HtoLi);%+[-15;15;-30];
    transmat=Lispm.mat;
    transmat(1:3,4)=-(Lispm.dim.*[Lispm.mat(1,1),Lispm.mat(2,2),Lispm.mat(3,3)])/2;
end