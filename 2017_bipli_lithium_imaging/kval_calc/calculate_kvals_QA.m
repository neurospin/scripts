%k-value de SSFP (TPI)

TR_SSFP=200;
TE_SSFP=0.3;

TEinclud=0;

alpha_SSFP=21;
T1=12500;
T2=2100;
T2star=12;
%Sigdivrho=(2*10^-5)/50;
%Sigdivrho=(1.4729*10^-6)/50;
Sigdivrho=(3.8)/100; %For B0 correct
%Sigdivrho= 11/50; %for B0 non corrected
kval_SSFP_val=kval_SSFP(Sigdivrho,alpha_SSFP,TR_SSFP,TE_SSFP,T1,T2,T2star);
kval_SSFP_val_2=kval_bSSFP(Sigdivrho,alpha_SSFP,TR_SSFP,TE_SSFP,T1,T2,T2star);

%k-value de trufi

%sig=2500/1000;
%sig=1148/100;
sig=29.5;
TR_bSSFP=5;
TE_bSSFP=2.5;
alpha_bSSFP=30;
T2star=12;
Sigdivrho=(sig)/100;

kval_bSSFP_val=kval_bSSFP(Sigdivrho,alpha_bSSFP,TR_bSSFP,TE_bSSFP,T1,T2,T2star);
%kval_bSSFP_val_2=kval_SSFP(Sigdivrho,alpha_bSSFP,TR_bSSFP,TE_bSSFP,T1,T2,T2star);

T1=3947.000;
T2=63.000;
T2star=5;

multiplier_SSFP=Im_to_rho_SSFP(kval_SSFP_val,alpha_SSFP,TR_SSFP,TE_SSFP,T1,T2,T2star);
multiplier_bSSFP=Im_to_rho_bSSFP(kval_bSSFP_val,alpha_bSSFP,TR_bSSFP,TE_bSSFP,T1,T2,T2star);

%T1=14000.000;
%T2=1670.000;
%T2star=5;
T1=12000;
T2=2100;

multiplier_SSFP_2=Im_to_rho_SSFP(kval_SSFP_val,alpha_SSFP,TR_SSFP,TE_SSFP,T1,T2,T2star);
multiplier_bSSFP_2=Im_to_rho_bSSFP(kval_bSSFP_val,alpha_bSSFP,TR_bSSFP,TE_bSSFP,T1,T2,T2star);
