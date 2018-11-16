
function bssfp_multip = Im_to_rho_bSSFP(kval,alpha,TR,T1,T2)

    E1=exp(-TR/T1);
    E2=exp(-TR/T2);
    alpharad=alpha*pi/180;
    %Im_to_rho_trufi=1/(kval*(tan(alpharad/2)*(1-(E1-cos(alpharad))*r)));
    bssfp_multip=(1-(E1-E2)*cos(alpharad)-E1*E2)/(kval*sqrt(E2*(1-E1)*sin(alpharad)));