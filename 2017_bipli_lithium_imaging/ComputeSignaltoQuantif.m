function ComputeSignaltoQuantif(filepath,alpha,Computedniipath,seq,mask_liquid,mask_brainmatter,B0cor,relax_params)

    if strcmp(seq,'TPI')
        
        %mask_brainmatter=spm_vol('C:\Users\js247994\Documents\Bipli2\PartialVQuantif_tests\2018_06_01\Maskstest\TPI_brainmatter.nii');
        %mask_liquid=spm_vol('C:\Users\js247994\Documents\Bipli2\PartialVQuantif_tests\2018_06_01\Maskstest\TPI_liquid.nii');        
        if B0cor
            %kval=0.1971;
            %kval=0.2562;
            kval=0.202;
        else
            kval=1.7036;
        end
        TR=200.000;
        TE=0.300;
    elseif strcmp(seq,'trufi')
        %mask_brainmatter=spm_vol('C:\Users\js247994\Documents\Bipli2\PartialVQuantif_tests\2018_06_01\Maskstest\trufi_brainmatter.nii');
        %mask_liquid=spm_vol('C:\Users\js247994\Documents\Bipli2\PartialVQuantif_tests\2018_06_01\Maskstest\trufi_liquid.nii');
        %kval=0.0565;
        %kval=0.0250;
        %kval=1.3515;
        %kval=1.0786;
        kval=1.4097;
        TR=5.000;
        TE=2.500;
        kval=kval*1000;
    end

    mask_liquid_spm=spm_vol(mask_liquid);
    mask_brainmatter_spm=spm_vol(mask_brainmatter);
    liquid_mat=spm_read_vols(mask_liquid_spm);
    brainmatter_mat=spm_read_vols(mask_brainmatter_spm);
    %res= 15;
    %T1=3947.000;
    %T2=63.000;
    %E1=exp(-TR/T1);
    %E2=exp(-TR/T2);
    
    TEinclud=0;
    if strcmp(seq,'TPI')
        multiplier_brainmatter=Im_to_rho_SSFP(kval,str2double(alpha),TR,TE,relax_params.brainmatter.T1,relax_params.brainmatter.T2,relax_params.brainmatter.T2star);
        multiplier_liquid=Im_to_rho_SSFP(kval,str2double(alpha),TR,TE,relax_params.liquid.T1,relax_params.liquid.T2,relax_params.brainmatter.T2star);
    elseif strcmp(seq,'trufi')
        multiplier_brainmatter=Im_to_rho_bSSFP(kval,str2double(alpha),TR,TE,relax_params.brainmatter.T1,relax_params.brainmatter.T2,relax_params.brainmatter.T2star);
        multiplier_liquid=Im_to_rho_bSSFP(kval,str2double(alpha),TR,TE,relax_params.liquid.T1,relax_params.liquid.T2,relax_params.liquid.T2star);
    end
    
    im_spm=spm_vol(filepath);
    im_mat=spm_read_vols(im_spm);
    im_size=size(im_mat);
    im_quantif_mat=zeros(im_size);
    im_masked_mat=zeros(im_size);
    im_uniform_mat=zeros(im_size);
    %brainmatsig=im_quantif_mat;
    %liquidsig=im_quantif_mat;
    brainmatter_mat(brainmatter_mat<0)=0;
    brainmatter_mat(brainmatter_mat>1)=1;
    liquid_mat(liquid_mat<0)=0;
    liquid_mat(liquid_mat>1)=1;
    brainper_mat=zeros(im_size);
    liquidper_mat=zeros(im_size);
    for i=1:im_size(1)
        for j=1:im_size(2)
            for k=1:im_size(3)
                %if mask[i,j,k]>0:
                    %%%% sig =[Litissue]*proport_tissue*multipliertissue_rho_to_im + [Librainmatter]* proport_brainmatter*multipliermatter_rho_to_im
                    %%%% if [Litissue]==[Librainmatter]
                    %%%% sig= [Li]*(proport_tiss*multip_tissue_rhotoim + proport_brainmatter*multip_matter_rhotoim
                    %%%% multip_mattter_rhotoim= 1/multip_matter_im_to_rho (calculated above)
                brainper=brainmatter_mat(i,j,k)/(brainmatter_mat(i,j,k)+liquid_mat(i,j,k));
                liquidper=liquid_mat(i,j,k)/(brainmatter_mat(i,j,k)+liquid_mat(i,j,k));
                brainper_mat(i,j,k)=brainper;
                liquidper_mat(i,j,k)=liquidper;
                
                if im_mat(i,j,k)<0
                    im_mat(i,j,k)=0;
                end                
                if brainmatter_mat(i,j,k)>0 || liquid_mat(i,j,k)>0
                    %im_quantif_mat(i,j,k)=im_mat(i,j,k)/(brainmatter_mat(i,j,k)*(1/multiplier_brainmatter)+liquid_mat(i,j,k)*(1/multiplier_liquid));
                    im_quantif_mat(i,j,k)=im_mat(i,j,k)/(brainper*(1/multiplier_brainmatter)+liquidper*(1/multiplier_liquid));
                elseif brainmatter_mat(i,j,k)<=0 && liquid_mat(i,j,k)<=0
                    im_quantif_mat(i,j,k)=im_mat(i,j,k)*multiplier_liquid;
                else
                    display('hi');
                end
                
                if im_quantif_mat(i,j,k)>3
                    display('too high');
                elseif im_quantif_mat(i,j,k)<0
                    display('too low');
                end
                
                im_uniform_mat(i,j,k)=im_mat(i,j,k)*multiplier_brainmatter;
                im_masked_mat(i,j,k)=im_quantif_mat(i,j,k)*(brainmatter_mat(i,j,k)+liquid_mat(i,j,k));
                %im_quantif_mat(i,j,k)
                %im_test_mat(i,j,k)
            end
        end
    end

    [folder,filename,ext]=fileparts(Computedniipath);
    Computedniipathuniform=fullfile(folder,filename+"_uniform"+ext);
    Computedniipathmasked=fullfile(folder,filename+"_masked"+ext);
    
    quantif_spm=im_spm;
    quantif_spm.fname=char(Computedniipath);
    uniform_spm=im_spm;
    uniform_spm.fname=char(Computedniipathuniform);
    quantif_spm.dt=[64 0];
    uniform_spm.dt=[64 0];

    masked_spm=im_spm;
    masked_spm.fname=char(Computedniipathmasked);
    masked_spm.dt=[64 0];   

    spm_write_vol(quantif_spm,im_quantif_mat);
    spm_write_vol(uniform_spm,im_uniform_mat);
    spm_write_vol(masked_spm,im_masked_mat);