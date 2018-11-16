
function ssfp_multip = Im_to_rho_SSFP(kval,alpha,TR,T1,T2)

E1=exp(-TR/T1);
E2=exp(-TR/T2);
alpharad=alpha*pi/180;
p=1-E1*cos(alpharad)-E2*E2*(E1-cos(alpharad));
q=E2*(1-E1)*(1+cos(alpha));
sqrtpq=sqrt(p*p-q*q);
r=(1-E2*E2)/(sqrtpq);
ssfp_multip=1/(kval*(tan(alpharad/2)*(1-(E1-cos(alpharad))*r)));