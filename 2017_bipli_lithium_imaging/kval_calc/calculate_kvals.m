%k-value de SSFP (TPI)

TR_SSFP=0.2;
TE_SSFP=0.0003;

alpha_SSFP=21;
T1=4.07;
T2=1.67;
T2star=0.012;
%Sigdivrho=(2*10^-5)/50;
%Sigdivrho=(1.4729*10^-6)/50;
Sigdivrho=(1.75)/50; %For B0 correct
%Sigdivrho= 11/50; %for B0 non corrected
kval_SSFP_val=kvalSSFP(Sigdivrho,alpha_SSFP,TR_SSFP,T1,T2);

%k-value de trufi

sig=2600/100;
%sig=1148/100;
TR_bSSFP=0.005;
TE_bSSFP=0.0003;
alpha=30;
T2star=0.012;
Sigdivrho=(sig)/50;

kval_bSSFP_val=kval_bSSFP(Sigdivrho,alpha_bSSFP,TR,T1,T2);

T1=3947.000;
T2=63.000;

multiplier_SSFP=Im_to_rho_SSFP(kval_SSFP_val,alpha_SSFP,TR_SSFP,T1,T2);
multiplier_bSSFP=Im_to_rho_bSSFP(kval_bSSFP_val,alpha_bSSFP,TR_bSSFP,T1,T2);
