# -*- coding:Utf-8 -*-
#
# Author : Arthur Coste
# Date : March 2015
# Purpose : Process Bruker Raw Data File 
#---------------------------------------------------------------------------------------------------------------------------------
# command : ProcessBrukerData.py --i E:\meas_MID86_ute_tra_TE100us_1H_1SL20mm_FID8353.dat --verbose --regrid=rad --vis=True 

import os,sys
import argparse, numpy
from ReadRawData import *
from Regridding import *
from nipy import save_image
from ctypes import *
STD_OUTPUT_HANDLE_ID = c_ulong(0xfffffff5)
windll.Kernel32.GetStdHandle.restype = c_ulong
std_output_hdl = windll.Kernel32.GetStdHandle(STD_OUTPUT_HANDLE_ID)

parser = argparse.ArgumentParser(epilog="RawDaRec (RawDataReconstruction) version 1.0")
parser.add_argument("--v","--verbose", help="output verbosity", action="store_true")
parser.add_argument("--filepath", type=str,help="path to sequence files directory")
parser.add_argument("--HeaderOnly",help="Only print Header information",action="store_true")
parser.add_argument("--o", type=str,help="Output file path and name (as NIfTI)")
parser.add_argument("--s","--save",help="Save Reconstructed image (with --o)",action="store_true")
parser.add_argument("--FISTA_CSV", help="Save Data as FISTA readable CSV Files", action = "store_true")
args = parser.parse_args()

if args.v: verbose=True
else: verbose = False

if not args.filepath :
	windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 12)
	print 'ERROR   : Path to sequence files not specified'
	windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
	sys.exit()
if args.filepath : filepath = args.filepath

if not os.path.exists(filepath) or not os.path.isdir(filepath):
	windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 12)
	print 'ERROR   : ERROR INCORECT PATH'
	windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
	sys.exit()

print 'Searching ', filepath, 'for files '
METHODsourcefile = os.path.join(filepath,"method")
FIDsourcefile = os.path.join(filepath,"fid")
TRAJsourcefile = os.path.join(filepath,"traj")

windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 10)
print '[FOUND] ', METHODsourcefile
print '[FOUND] ', FIDsourcefile
print '[FOUND] ', TRAJsourcefile
windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)

if args.HeaderOnly:
	HeaderOnly=True
else:
	HeaderOnly=False
	
if args.s and args.o:
	if args.o.lower().endswith('.nii'):
		Save=True
		OutputPath=args.o
	else:
		Save=False
		OutputPath=None
	
if args.FISTA_CSV:
	FISTA_CSV=True
else :
	FISTA_CSV=False	
	
print
print '------------------------------------------------------------'
print 'Processing Pipeline :'
if HeaderOnly:
	print '\t\t\t => Extracting Header Information Only'

if HeaderOnly: 
	BrukerRawData(FIDsourcefile,METHODsourcefile,TRAJsourcefile,HeaderOnly)
else : 
	CplxData,Kx,Ky,Kz,parameters,coefsR,coefsP,coefsS = BrukerRawData(FIDsourcefile,METHODsourcefile,TRAJsourcefile,HeaderOnly)
windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 10) 
if HeaderOnly: print 'Header Information Extracted'
else : print 'Data and parameters extracted'
windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)

if FISTA_CSV:
	windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 15)
	print 'INFO    : Writing CSV Files for FISTA'
	print CplxData.shape
	windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
	print Kx.shape

	FISTApath = os.path.dirname(OutputPath)
	FISTApath = os.path.join( FISTApath, 'FISTA_BrukerKspaceValues.csv' )
	if not os.path.isfile(FISTApath) :
		# f=open("TPI_Kspace_positions_values_31P_bouleFawziTubes_p075_40960proj_FA10_TE4_5.csv","w")
		print 'INFO    : Writing CSV file for Bruker Data '
		f=open(FISTApath,"w")
		#####f.write(float(KX))
		### en changeant l'incrément de la boucle on retourne plus ou moins de points de la trajectoire ! faire attention !!
		# for i in range(0,KX.shape[0],100):
		# echo = 0;
		# for coil in range(1):
		for i in range(Kx.shape[0]):
			# for j in range(0,KX.shape[1],10):
				f.write(str(1))
				f.write(',')
				f.write(str(1))
				f.write(',')
				f.write(str(i))
				f.write(',')
				f.write(str(float(Kx[i])))
				f.write(',')
				f.write(str(float(Ky[i])))
				f.write(',')
				f.write(str(float(Kz[i])))
				f.write(',')
				f.write(str(numpy.real(CplxData[i%CplxData.shape[0],i%48])))
				f.write(',')
				f.write(str(numpy.imag(CplxData[i%CplxData.shape[0],i%48])))
				f.write("\n")
		f.close()



if not HeaderOnly:
	ReconstructedImg,modulekspace,phasekspace=RegridBruker(parameters[0],parameters[1],parameters[2],1,1,parameters[3],CplxData,Kx,Ky,Kz,coefsR,coefsP,coefsS,parameters[10],verbose)
	from nifty_funclib import SaveArrayAsNIfTI,SaveArrayAsNIfTI_2
	SaveArrayAsNIfTI(ReconstructedImg,float(parameters[7]),float(parameters[8]),float(parameters[9]),OutputPath)
	SaveArrayAsNIfTI(modulekspace,1,1,1,"test_fantome_bruker_Kspace_Hsymetry2_test.nii")
	SaveArrayAsNIfTI(phasekspace,1,1,1,"test_fantome_bruker_Kspace_Hsymetry2_test_phase.nii")
