%k-value de SSFP (TPI)

TR_SSFP=200;
TE_SSFP=0.3;

TEinclud=1;

alpha_SSFP=21;
T1=5000;
T2=60;
T2star=0.012;
%Sigdivrho=(2*10^-5)/50;
%Sigdivrho=(1.4729*10^-6)/50;
Sigdivrho=(2.5)/100; %For B0 correct
%Sigdivrho= 11/50; %for B0 non corrected
kval_SSFP_val=kval_SSFP(Sigdivrho,alpha_SSFP,TR_SSFP,TE_SSFP,T1,T2,TEinclud);
kval_SSFP_val_2=kval_bSSFP(Sigdivrho,alpha_SSFP,TR_SSFP,TE_SSFP,T1,T2,TEinclud);

%k-value de trufi

sig=46;
%sig=1148/100;
TR_bSSFP=5;
TE_bSSFP=0.3;
alpha_bSSFP=30;
T2star=0.012;
Sigdivrho=(sig)/100;

kval_bSSFP_val=kval_bSSFP(Sigdivrho,alpha_bSSFP,TR_bSSFP,TE_bSSFP,T1,T2,TEinclud);
kval_bSSFP_val_2=kval_SSFP(Sigdivrho,alpha_bSSFP,TR_bSSFP,TE_bSSFP,T1,T2,TEinclud);

T1=3947.000;
T2=63.000;

multiplier_SSFP=Im_to_rho_SSFP(kval_SSFP_val,alpha_SSFP,TR_SSFP,TE_SSFP,T1,T2,TEinclud);
multiplier_bSSFP=Im_to_rho_bSSFP(kval_bSSFP_val,alpha_bSSFP,TR_bSSFP,TE_bSSFP,T1,T2,TEinclud);

T1=14000.000;
T2=1500.000;

multiplier_SSFP_2=Im_to_rho_SSFP(kval_SSFP_val,alpha_SSFP,TR_SSFP,TE_SSFP,T1,T2,TEinclud);
multiplier_bSSFP_2=Im_to_rho_bSSFP(kval_bSSFP_val,alpha_bSSFP,TR_bSSFP,TE_bSSFP,T1,T2,TEinclud);

%multiplier_SSFP_2=Im_to_rho_bSSFP(kval_SSFP_val_2,alpha_SSFP,TR_SSFP,TE_SSFP,T1,T2,TEinclud);
%multiplier_bSSFP_2=Im_to_rho_SSFP(kval_bSSFP_val_2,alpha_bSSFP,TR_bSSFP,TE_bSSFP,T1,T2,TEinclud);