function ComputeSignaltoQuantif(filepath,alpha,Computedniipath,Computedniipathtest,seq,B0cor,relax_params)

    if strcmp(seq,'TPI')
        
        mask_brainmatter=spm_vol('C:\Users\js247994\Documents\Bipli2\PartialVQuantif_tests\2018_06_01\Maskstest\TPI_brainmatter.nii');
        mask_liquid=spm_vol('C:\Users\js247994\Documents\Bipli2\PartialVQuantif_tests\2018_06_01\Maskstest\TPI_liquid.nii');
        brainmatter_mat=spm_read_vols(mask_brainmatter);
        liquid_mat=spm_read_vols(mask_liquid);
        if B0cor
            kval=0.1971;
        else
            kval=1.2387;
        end
        TR=200.000;
        TE=0.300;
    elseif strcmp(seq,'trufi')
        mask_brainmatter=spm_vol('C:\Users\js247994\Documents\Bipli2\PartialVQuantif_tests\2018_06_01\Maskstest\trufi_brainmatter.nii');
        mask_liquid=spm_vol('C:\Users\js247994\Documents\Bipli2\PartialVQuantif_tests\2018_06_01\Maskstest\trufi_liquid.nii');
        brainmatter_mat=spm_read_vols(mask_brainmatter);
        liquid_mat=spm_read_vols(mask_liquid);
        kval=0.0565;
        TR=5.000;
        TE=2.500;
        kval=kval*1000;
    end

    %res= 15;
    %T1=3947.000;
    %T2=63.000;
    %E1=exp(-TR/T1);
    %E2=exp(-TR/T2);
    
    TEinclud=1;
    if strcmp(seq,'TPI')
        multiplier_brainmatter=Im_to_rho_SSFP(kval,str2double(alpha),TR,TE,relax_params.brainmatter.T1,relax_params.brainmatter.T2,TEinclud);
        multiplier_liquid=Im_to_rho_SSFP(kval,str2double(alpha),TR,TE,relax_params.liquid.T1,relax_params.liquid.T2,TEinclud);
    elseif strcmp(seq,'trufi')
        multiplier_brainmatter=Im_to_rho_bSSFP(kval,str2double(alpha),TR,TE,relax_params.brainmatter.T1,relax_params.brainmatter.T2,TEinclud);
        multiplier_liquid=Im_to_rho_bSSFP(kval,str2double(alpha),TR,TE,relax_params.liquid.T1,relax_params.liquid.T2,TEinclud);
    end
    
    im_spm=spm_vol(filepath);
    im_mat=spm_read_vols(im_spm);
    im_size=size(im_mat);
    im_quantif_mat=zeros(im_size);
    im_test_mat=zeros(im_size);
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
                if brainmatter_mat(i,j,k)>0 || liquid_mat(i,j,k)>0
                    %im_quantif_mat(i,j,k)=im_mat(i,j,k)/(brainmatter_mat(i,j,k)*(1/multiplier_brainmatter)+liquid_mat(i,j,k)*(1/multiplier_liquid));
                    im_quantif_mat(i,j,k)=im_mat(i,j,k)/(brainper*(1/multiplier_brainmatter)+liquidper*(1/multiplier_liquid));
                elseif brainmatter_mat(i,j,k)<=0 && liquid_mat(i,j,k)<=0
                    im_quantif_mat(i,j,k)=im_mat(i,j,k)*multiplier_liquid;
                else
                    display('hi');
                end
                im_test_mat(i,j,k)=im_mat(i,j,k)*multiplier_brainmatter;
                if im_quantif_mat(i,j,k)>3
                    display('too high');
                elseif im_quantif_mat(i,j,k)<0
                    display('too low');
                end
                %im_quantif_mat(i,j,k)=im_quantif_mat(i,j,k)*(brainmatter_mat(i,j,k)+liquid_mat(i,j,k));
                %im_quantif_mat(i,j,k)
                %im_test_mat(i,j,k)
            end
        end
    end
                                        
     quantif_spm=im_spm;
     quantif_spm.fname=char(Computedniipath);
     test_spm=im_spm;
     test_spm.fname=char(Computedniipathtest);
     quantif_spm.dt=[64 0];
     test_spm.dt=[64 0];
     spm_write_vol(quantif_spm,im_quantif_mat);
     spm_write_vol(test_spm,im_test_mat);