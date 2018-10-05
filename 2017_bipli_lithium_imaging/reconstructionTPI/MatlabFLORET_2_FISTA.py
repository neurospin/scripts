## Process FLORET Matlab csv Files to have them in my FISTA format

import os,sys
import numpy

print("-------------------------------------------------------------")
print("Processing Files")
# Kpos=open("C:\\Users\\AC243636\\Documents\\FLORET\\csv_files\\NewSandroFev3\\ktraj.csv",'r')
# Kpos = open("E:\\Experiments_DATA\\Sodium\\Sodium_Aachen_1_03_16\\Data_FISTA\\Acq1\\Ktraj.csv",'r')
Kpos = open("Z:\\arthur_coste\\ToBeProcessedMatlabLaptop\\FloretKspace\\ktraj.csv",'r')


ind=0
# 300 spokes 1373 points
# Sandro's july tests 279 spokes of 1676 points

# SaltIT pipeline has 300 spokes of 1561 points
for line in Kpos:
        # print ind
        if (ind==0):
            x=line.split(',')    
        if (ind==1):
            y=line.split(',')
        if (ind==2):
            z=line.split(',')
        ind=ind+1
print("number of points in trajectory = ", int(len(x)))

Kpos.close()
kdatan=[]
# Kdata=open("C:\\Users\\AC243636\\Documents\\FLORET\\csv_files\\NewSandroFev3\\rawdata.csv",'r')
# Kdata=open("E:\\Experiments_DATA\\Sodium\\Sodium_Aachen_1_03_16\\Data_FISTA\\Acq1\\rawdata.csv",'r')
Kdata=open("Z:\\arthur_coste\\ToBeProcessedMatlabLaptop\\FloretKspace\\rawdata.csv",'r')
# kdata=[]
for line in Kdata:
            kdata=line.split(',')
            nb_spokes = int(len(kdata))
            for pts in kdata:
                kdatan.append(pts.split(','))
kdatac=[]                
for i in range(300):    
    a= kdatan[i::300]
    for j in range(len(a)):
        kdatac.append(a[j])    

print("number of values in trajectory = ", int(len(kdatan)))
Kdata.close()        
print("Writting Fista readable File")

coil=1
# exponent=0
# for i in range(len(kdatan)):
    # a=str(kdatan[i])
    # print a
    # print (int(len(a)))
    # for i in range(int(len(a))):
        # if str(a[i]) == '-':
            # signepos=i
        # if str(a[i]) == '+':
            # signepos=i
        # if str(a[i]) == 'e':
            # exponent=i            
    # print signepos    
    # print exponent
    # if exponent==0 :
        # print a[2:signepos]
        # print a[signepos+1:len(a)-3]
    # if exponent!=0 :
        # print a[2:signepos]
        # print a[signepos+1:exponent+2]
# if os.path.isfile("E:\\Experiments_DATA\\Sodium\\Sodium_Aachen_1_03_16\\Data_FISTA\\Acq1\\FLORET_posval_Sodium_invivo.csv") :
if os.path.isfile("Z:\\arthur_coste\\ToBeProcessedMatlabLaptop\\FloretKspace\\FLORET_posval_Sodium_invivo.csv") :
    print("File already exists !")
# if not os.path.isfile("E:\\Experiments_DATA\\Sodium\\Sodium_Aachen_1_03_16\\Data_FISTA\\Acq1\\FLORET_posval_Sodium_invivo.csv") :
if not os.path.isfile("Z:\\arthur_coste\\ToBeProcessedMatlabLaptop\\FloretKspace\\FLORET_posval_Sodium_invivo.csv") :
    f=open("Z:\\arthur_coste\\ToBeProcessedMatlabLaptop\\FloretKspace\\FLORET_posval_Sodium_invivo.csv","w")
    for point in range(int(len(kdatac))):
    # print "spoke ", int(round(point/nb_spokes)), "point " , int(point), x[point], y[point] , z[point]
        f.write(str(coil))
        f.write(',')
        f.write(str(int(round(point/1561))))
        f.write(',')
        f.write(str(point))
        f.write(',')
        f.write(str(float(x[point])))
        f.write(',')
        f.write(str(float(y[point])))
        f.write(',')
        f.write(str(float(z[point])))
        f.write(',')
        a = str(kdatac[point])
        f.write(str(a[2:len(kdatac[point])-3]))
        f.write("\n")    
    f.close()        
# print "[DONE] --> E:\\Experiments_DATA\\Sodium\\Sodium_Aachen_1_03_16\\Data_FISTA\\Acq1\\FLORET_posval_Sodium_invivo.csv"
print("[DONE] --> Z:\\arthur_coste\\ToBeProcessedMatlabLaptop\\FloretKspace\\FLORET_posval_Sodium_invivo.csv")
print("-------------------------------------------------------------")