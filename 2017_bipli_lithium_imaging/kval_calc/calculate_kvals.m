%k-value de SSFP (TPI)
%ALL data presented here based on acquisitions taken on 2018/10/25 on
%50mmol cylinder
TEinclud=0;

TR_SSFP=200;
TE_SSFP=0.3;

alpha_SSFP=21;
%T1=4070;
%T2=600;
T1=5000;
T2=1900;
T2star=12;
Sigdivrho=2/50; % B0 correction, Density Compensation Second Line
%Sigdivrho=(12.3)/50; % B0 correction, Density Compensation First line
%Sigdivrho=(0.189)/50; % Sandro reconstruct, no filter, no B0
%Sigdivrho=(12.1)/50; % No B0 correction, Density Compensation First line
%Sigdivrho=(1.75)/50; %For B0 correct
%Sigdivrho=(2.22)/50; %For B0 correct
%Sigdivrho= 11/50; %for B0 non corrected
%Sigdivrho=(2*10^-5)/50;
%Sigdivrho=(1.4729*10^-6)/50;
kval_SSFP_val=kval_SSFP(Sigdivrho,alpha_SSFP,TR_SSFP,TE_SSFP,T1,T2,T2star);
%kval_SSFP_val_2=kval_bSSFP(Sigdivrho,alpha_SSFP,TR_SSFP,TE_SSFP,T1,T2,TEinclud);

%k-value de trufi

%sig=2600/100;  %% to get original val for k-multip, this would need to be sig=3760/100;
sig=1550/100;
TR_bSSFP=5;
TE_bSSFP=2.5;
alpha_bSSFP=29;
T2star=20;
Sigdivrho=(sig)/50;

kval_bSSFP_val=kval_bSSFP(Sigdivrho,alpha_bSSFP,TR_bSSFP,TE_bSSFP,T1,T2,T2star);
kval_bSSFP_val_SSFP=kval_SSFP(Sigdivrho,alpha_bSSFP,TR_bSSFP,TE_bSSFP,T1,T2,T2star);

T1=3947.000;
T2=68.000;
T2star=30;
alpha_bSSFP=30;

%kval_bSSFP_val=kval_bSSFP_val*1000;
multiplier_SSFP=Im_to_rho_SSFP(kval_SSFP_val,alpha_SSFP,TR_SSFP,TE_SSFP,T1,T2,T2star);
multiplier_bSSFP=Im_to_rho_bSSFP(kval_bSSFP_val,alpha_bSSFP,TR_bSSFP,TE_bSSFP,T1,T2,T2star);
multiplier_bSSFP_SSFP=Im_to_rho_SSFP(kval_bSSFP_val_SSFP,alpha_bSSFP,TR_bSSFP,TE_bSSFP,T1,T2,T2star);

T1=14000.000;
T2=1670.000;

T1=5000;
T2=1900;

multiplier_SSFP_2=Im_to_rho_SSFP(kval_SSFP_val,alpha_SSFP,TR_SSFP,TE_SSFP,T1,T2,T2star);
multiplier_bSSFP_2=Im_to_rho_bSSFP(kval_bSSFP_val,alpha_bSSFP,TR_bSSFP,TE_bSSFP,T1,T2,T2star);

%multiplier_SSFP_2=Im_to_rho_bSSFP(kval_SSFP_val_2,alpha_SSFP,TR_SSFP,TE_SSFP,T1,T2,TEinclud);
%multiplier_bSSFP_2=Im_to_rho_SSFP(kval_bSSFP_val_2,alpha_bSSFP,TR_bSSFP,TE_bSSFP,T1,T2,TEinclud);