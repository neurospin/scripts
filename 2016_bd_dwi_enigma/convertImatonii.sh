#!/bin/sh

for mydir in */ ; do
    #dcm2nii -4 ${mydir}/DTI/*.IMA
    #mkdir IMAfiles
    mv ${mydir}/DTI/*.IMA IMAfiles
done

