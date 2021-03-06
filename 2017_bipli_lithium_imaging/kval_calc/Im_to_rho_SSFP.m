
function ssfp_multip = Im_to_rho_SSFP(kval,alpha,TR,TE,T1,T2,T2star)

E1=exp(-TR/T1);
E2=exp(-TR/T2);
%E22=exp(-TE/T2);
alpharad=alpha*pi/180;
p=1-E1*cos(alpharad)-E2*E2*(E1-cos(alpharad));
q=E2*(1-E1)*(1+cos(alpharad));
sqrtpq=sqrt(p*p-q*q);
r=(1-E2*E2)/(sqrtpq);
SSFP_sig=SSFPsig_calc(alpha,T1,T2,T2star,TR,TE);
ssfp_multip=1/(SSFP_sig*kval);
%if TEinclud
%    ssfp_multip=1/(E22*kval*(tan(alpharad/2)*(1-(E1-cos(alpharad))*r)));
%else
%ssfp_multip=1/(kval*(tan(alpharad/2)*(1-(E1-cos(alpharad))*r)));
%end