
function [kvalSSFP_val] = kval_SSFP(Sigdivrho,alpha,TR,TE,T1,T2,T2star)
    alpharad=alpha*pi/180;
    E1=exp(-TR/T1);
    E2=exp(-TR/T2);
    E22=exp(-TE/T2);
    %E2star=exp(-TE/T2);
    p=1-E1*cos(alpharad)-E2*E2*(E1-cos(alpharad));
    q=E2*(1-E1)*(1+cos(alpha));
    sqrtpq=sqrt(p*p-q*q);
    r=(1-E2*E2)/(sqrtpq);
    SSFP_sig=SSFPsig_calc(alpha,T1,T2,T2star,TR,TE);
    kvalSSFP_val=Sigdivrho/SSFP_sig;
    %if TEinclud
        %kvalSSFP_val=(Sigdivrho)/(tan(alpharad/2)*(1-(E1-cos(alpharad))*r)*E22);
    %else
    %kvalSSFP_val=(Sigdivrho)/(tan(alpharad/2)*(1-(E1-cos(alpharad))*r));
    %end
end
