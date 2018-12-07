function bSSFPsig = bSSFPsig_calc(alpha,T1,T2,T2star,TR,TE)

    alpharad=alpha*pi/180;
    E1=exp(-TR/T1);
    E2=exp(-TR/T2);
    %E22=exp(-TE/T2);
    bSSFPsig=sin(alpharad)*((1-E1)/(1-(E1-E2)*cos(alpharad)-E1*E2))*exp(-TE/T2);