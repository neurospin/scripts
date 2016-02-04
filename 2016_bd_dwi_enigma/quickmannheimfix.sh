#!/bin/sh

for mydir in mannheim2/*/ ; do
    #dcm2nii -4 ${mydir}/DTI/*.IMA
    #mkdir IMAfiles
    mv mannheim2/${mydir}/DTI/*.IMA mannheim/*{mydir}/DTI/IMAfiles
done


