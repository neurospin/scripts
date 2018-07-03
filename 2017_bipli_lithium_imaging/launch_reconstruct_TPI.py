# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 19:45:20 2018

@author: JS247994
"""

#function launch_reconstruct_TPI(dirdat,ProcessedTPIpath,subjectnumber,reconstructfile,Pythonexe )

#launch_reconstruct_TPI('C:\Users\js247994\Documents\Bipli2\Test8\Raw','C:\Users\js247994\Documents\Bipli2\Test8\Processed','C:\Users\js247994\Documents\Bipli2\BipliPipeline\scripts\2017_bipli_lithium_imaging\ReconstructionTPI','C:\Python27\python.exe')

import os #, sys
#import openpyxl
#import argparse
import numpy as np
import subprocess

def launch_reconstruct(dirdat,ProcessedTPIpath,subjectnumber,reconstructfile):
    listdat=os.listdir(dirdat);
    
    for j in range(1,np.size(listdat)):
        ok=listdat[j].find('TPI');
        if ok>-1:
            Tpifilename=listdat[j]
            Tpifilepath=os.path.join(dirdat,Tpifilename);
            deg=Tpifilename.find('deg');
            degval=(Tpifilename[deg-2:deg-1]);
            TPIresultname=('Patient'+str(subjectnumber)+'_'+degval+'deg.nii');
            Reconstructpath=os.path.join(ProcessedTPIpath,(TPIresultname));
            codelaunch=({'python2 '}+reconstructfile+{' --i '}+Tpifilepath+{' --NSTPI --s --FISTA_CSV --o '}+Reconstructpath);
            subprocess.run(codelaunch)
            #status = system(codelaunch);
  