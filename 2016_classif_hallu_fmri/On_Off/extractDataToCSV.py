# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:26:18 2015

@author: amicie
"""

import re
BASE_PATH ="/neurospin/brainomics/2016_classif_hallu_fmri/data/DATA_Localizer/Patients MMH/"


def sort_scans(path_prt):
    f=open(path_prt,'r')
    myList=[]
    
    for line in f:
        myList.append(line)
        
    if 'Off\r\n' in myList:       
        off_line=myList.index('Off\r\n')
        nperiods=int(myList[off_line +1])
    if 'off\r\n' in myList:    
        off_line=myList.index('off\r\n')
        nperiods=int(myList[off_line +1])
        
    if 'Off\n' in myList:       
        off_line=myList.index('Off\n')
        nperiods=int(myList[off_line +1])
        
    if 'off\n' in myList:    
        off_line=myList.index('off\n')
        nperiods=int(myList[off_line +1])    
        

    for i in range(2,nperiods+2):

        scans=np.arange(int(myList[off_line +i].split()[0]),int(myList[off_line +i].split()[1])+1)
        if i==2:
            off_scans=scans
        else:  
            off_scans=np.append(off_scans,scans)
            
    if 'Transi\r\n' in myList:       
        transi_line=myList.index('Transi\r\n')
        nperiods=int(myList[transi_line +1])
    if 'transi\r\n' in myList:    
        transi_line=myList.index('transi\r\n')
        nperiods=int(myList[transi_line +1])
        
    if 'Transi\n' in myList:       
        transi_line=myList.index('Transi\n')
        nperiods=int(myList[transi_line +1])
        
    if 'transi\n' in myList:    
        transi_line=myList.index('transi\n')
        nperiods=int(myList[transi_line +1])    
        

    for i in range(2,nperiods+2):

        scans=np.arange(int(myList[transi_line +i].split()[0]),int(myList[transi_line +i].split()[1])+1)
        if i==2:
            transi_scans=scans
        else:   
            transi_scans=np.append(transi_scans,scans)
    
    if 'On\r\n' in myList:       
       on_line=myList.index('On\r\n')
       nperiods=int(myList[on_line +1])
    if 'on\r\n' in myList:    
       on_line=myList.index('on\r\n')
       nperiods=int(myList[on_line +1])
        
    if 'On\n' in myList:       
       on_line=myList.index('On\n')
       nperiods=int(myList[on_line +1])
        
    if 'on\n' in myList:    
       on_line=myList.index('on\n')
       nperiods=int(myList[on_line +1])        
   
   

    for i in range(2,nperiods+2):

        scans=np.arange(int(myList[on_line +i].split()[0]),int(myList[on_line +i].split()[1])+1)
        if i==2:
            on_scans=scans
        else:   
            on_scans=np.append(on_scans,scans)
    
    if 'End\r\n' in myList :       
        end_line=myList.index('End\r\n')
        nperiods=int(myList[end_line +1])
        
    if 'end\r\n' in myList:    
       end_line=myList.index('end\r\n')
       nperiods=int(myList[end_line +1])
        
    if 'End\n' in myList:       
       end_line=myList.index('End\n')
       nperiods=int(myList[end_line +1])
        
    if 'end\n' in myList:    
       end_line=myList.index('end\n')
       nperiods=int(myList[end_line +1])
        

    for i in range(2,nperiods+2):

        scans=np.arange(int(myList[end_line +i].split()[0]),int(myList[end_line +i].split()[1])+1)
        if i==2:
            end_scans=scans
        else: 
            end_scans=np.append(end_scans,scans)
            
    return off_scans,on_scans,transi_scans,end_scans


#number_scans=len(end_scans)+len(on_scans)+len(off_scans)+len(transi_scans)
#



 
def extract_data(list_of_scans,i,state):
    path_subject=BASE_PATH +'Lil'+str(i)
    for ele in list_of_scans:
        if ele<10:
            numScan='000' + str(ele)
        if ele>9 and ele<100:
            numScan='00' + str(ele)
        if ele>99:
            numScan='0' + str(ele)
        for file in os.listdir(path_subject):
            if re.findall(numScan+'.nii', file):
                namescan= file
                path_scan=os.path.join(path_subject,namescan)
                subject='Lil'+str(i)
                current=[subject,numScan,path_scan,state]
                c.writerow(current)
      

          
import csv
import re
import os                 
                    
#c=csv.writer(open('/neurospin/brainomics/2016_classif_hallu_fmri/on-off_svm/patientsTEST.csv','wb'),delimiter=',')

f=open('/neurospin/brainomics/2016_classif_hallu_fmri/toward_on/patients.csv','wb')
c=csv.writer(f,delimiter=',')
c.writerow(["Subject","Time","Scan","State"])
                     
 
for i in range(1,31):
    imagefile_pattern = 'Lil'+str(i)+'_'
    for file in os.listdir(BASE_PATH):
        if re.match(imagefile_pattern, file):
            name= file
            print(name)
            path_prt=os.path.join(BASE_PATH,name)
            off_scans,on_scans,transi_scans,end_scans=sort_scans(path_prt)  
            extract_data(off_scans,i,'off')
            extract_data(on_scans,i,'transi')
f.close()
#           

       
            
#
#cr=csv.reader(open('/neurospin/brainomics/2016_classif_hallu_fmri/on-off_svm/patientsTEST.csv',"rb"))
#for row in cr:
#    print row
#    
    

