function kval_bSSFP_val = kval_bSSFP(Sigdivrho,alpha,TR,TE,T1,T2,T2star)
    alpharad=alpha*pi/180;
    E1=exp(-TR/T1);
    E2=exp(-TR/T2);
    E22=exp(-TE/T2);
    b_SSFP_sig=bSSFPsig_calc(alpha,T1,T2,T2star,TR,TE);
    kval_bSSFP_val=Sigdivrho/b_SSFP_sig;
    %if TEinclud
        %kval_bSSFP_val=(Sigdivrho*sqrt(1-(E1-E2)*cos(alpharad)-E1*E2))/(sqrt(E2*(1-E1)*sin(alpharad))*E22);
    %else
        %kval_bSSFP_val=(Sigdivrho*sqrt(1-(E1-E2)*cos(alpharad)-E1*E2))/(sqrt(E2*(1-E1)*sin(alpharad)));
    %    kval_bSSFP_val=(Sigdivrho*(1-(E1-E2)*cos(alpharad)-E1*E2))/((1-E1)*exp(-TE/T2));
    %end
    
    %sqrt(E2*(1-E1)*sin(alpharad)/(1-(E1-E2)*cos(alpharad)-E1*E2));
    %sin(alpharad)/(1+cos(alpharad)+(1-cos(alpharad))*(T1/T2))

end


%kvalSPGR=((Sigdivrho)*(1-(E1-E2)*cos(alpharad)-E1*E2))/((1-E1)*sin(alpharad));
%kvalSPGR1=((Sigdivrho)*(1-E1*cos(alpharad))/((1-E1)*sin(alpharad)*E2star));
%kvalSSFP_H=(Sigdivrho)/(tan(alpharad/2)*(1-(E1-cos(alpharad))*r)); % Hanicke 2003
%kval_bSSFP=(Sigdivrho*(1-(E1-E2)*cos(alpharad)-E1*E2))/(sqrt(E2)*(1-E1)*sin(alpharad));
%kval_bSSFP_trufi=(Sigdivrho)/(tan(alpharad/2)*(1-(E1-cos(alpharad))*r));