function SSFPsig = SSFPsig_calc(alpha,T1,T2,T2star,TR,TE)

    alpharad=alpha*pi/180;
    E1=exp(-TR/T1);
    E2=exp(-TR/T2);
    %E22=exp(-TE/T2);
    p=1-E1*cos(alpharad)-E2*E2*(E1-cos(alpharad));
    q=E2*(1-E1)*(1+cos(alpha));
    sqrtpq=sqrt(p*p-q*q);
    r=(1-E2*E2)/(sqrtpq);
    SSFPsig=(tan(alpharad/2)*(1-(E1-cos(alpharad))*r));