# -*- coding:Utf-8 -*-
#
# Author : Arthur Coste
# Date : December 2014
# Purpose : Process Siemens Raw Data File 
#---------------------------------------------------------------------------------------------------------------------------------
# command : ProcessData.py --i E:\meas_MID86_ute_tra_TE100us_1H_1SL20mm_FID8353.dat --verbose --regrid=rad --vis=True 

import os,sys
import argparse, numpy
from ReadRawData import *
from Regridding import *
from nipy import save_image
from ctypes import *
#STD_OUTPUT_HANDLE_ID = c_ulong(0xfffffff5)
#windll.Kernel32.GetStdHandle.restype = c_ulong
#std_output_hdl = windll.Kernel32.GetStdHandle(STD_OUTPUT_HANDLE_ID)

import time

start = time.time()

parser = argparse.ArgumentParser(epilog="RawDaRec (RawDataReconstruction) version 1.0")
parser.add_argument("--v","--verbose", help="output verbosity", action="store_true")
parser.add_argument("--i", type=str,help="Input file path")
parser.add_argument("--o", type=str,help="Output file path and name (as NIfTI)")
parser.add_argument("--vis",help="enable Visualization",action="store_true")
parser.add_argument("--regrid",type=str,help="Available trajectories : RAD")
parser.add_argument("--s","--save",help="Save Reconstructed image (with --o)",action="store_true")
parser.add_argument("--Interp",type=str,help="Interpolation Type for Radial Regridding (linear or cubic NOT nearest)")
parser.add_argument("--HeaderOnly",help="Only print Header information",action="store_true")
parser.add_argument("--TPI",help="Sandro's TPI trajectory ",action="store_true")
parser.add_argument("--Cart3D",help="3D Cartesian Trajectory",action="store_true")
parser.add_argument("--NSTPI",help="NS TPI trajectory ",action="store_true")
parser.add_argument("--Bydder",help="Enable Bydder Multicoil Reconstruction",action="store_true")
parser.add_argument("--ManualBW",help="Manualy specify BW for 2D UTE",action="store_true")
parser.add_argument("--SavePhase",help="Save Phase Image",action="store_true")
parser.add_argument("--FISTAfile", help = "Create FISTA compatible file", action="store_true")
parser.add_argument("--S", "--Spectre", help = "Process Spectroscopy Data", action="store_true")
parser.add_argument("--PSF_ND", help="Compute Point Spread Function  with NO DECAY", action="store_true")
parser.add_argument("--PSF_D", help="Compute PSF with T2 decay", action ="store_true")
parser.add_argument("--B1sensitivity", help="Compute Sensitivity map by massive subsampling of acquisition", action = "store_true")
parser.add_argument("--FISTA_CSV", help="Save Data as FISTA readable CSV Files", action = "store_true")
parser.add_argument("--SaveKspace", help="Export regridded Kspace", action = "store_true")
parser.add_argument("--UseFullDCF", help="Use Full DCF Computation --> LONG", action = "store_true")
parser.add_argument("--SumEchoes", help="Sum all echoes K space", action = "store_true")
parser.add_argument("--MIA", help="Metabolic Interleaved Acquisition", action = "store_true")
args = parser.parse_args()

if args.v: verbose=True
else: verbose = False
if args.vis: vis=True
else: vis=False
if not args.i :
	#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 12)
	print('ERROR   : Input file not specified')
	#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
	sys.exit()
if args.i : source_file = args.i
if args.regrid: 
	regrid= True
	regridding=str(args.regrid)
	if regridding.upper() == 'RAD':
		rad=True
	else :
		rad=False	
else: 
	regrid = False
	regridding=None
	Cartesian=True
	rad=False
if args.s and args.o:
	if args.o.lower().endswith('.nii'):
		Save=True
		OutputPath=args.o
	elif args.o.lower().endswith('\\'):
		Save=True
		OutputPath=os.path.join(args.o,os.path.basename(source_file))
	else:
		Save=False
		OutputPath=None
else:
	Save=False
	OutputPath=None
if args.Interp:
	Interpolation=str(args.Interp)
else:
	Interpolation='linear'
	
if args.HeaderOnly:
	HeaderOnly=True
else:
	HeaderOnly=False
if args.TPI:
	TPI=True
else:
	TPI=False
if args.NSTPI:
	TPI=True
	NS_TPI=True
else:
	NS_TPI=False
	
if args.Bydder:
	Bydder=True
else : 
	Bydder=False
if args.SavePhase:
	SavePhase=True
else:
	SavePhase=False
if args.Cart3D:
	Cart3D=True
else:
	Cart3D=False	
if args.FISTAfile:
	FISTAfile=True
else:
	FISTAfile=False
	
if args.S:
	Spectro=True
else:
	Spectro=False
if args.PSF_ND:
	PSF_ND=True
else :
	PSF_ND=False
	
if args.PSF_D:
	PSF_D=True
else :
	PSF_D=False	
	
if args.B1sensitivity:
	B1sensitivity=True
else :
	B1sensitivity=False

if args.FISTA_CSV:
	FISTA_CSV=True
else :
	FISTA_CSV=False
	
if args.SaveKspace:
	SaveKspace=True
else :
	SaveKspace=False

if args.UseFullDCF:
	UseFullDCF=True
else :
	UseFullDCF=False	

if args.MIA:
	MIA=True
else :
	MIA=False	

if args.SumEchoes:
	SumEchoes=True
else :
	SumEchoes=False	
print()
print('------------------------------------------------------------')
print('Processing Pipeline :')
if HeaderOnly and not Spectro:
	print('\t\t\t => Extracting Header Information Only')
if Spectro:
	print('\t\t\t => Read Raw Data')
	print('\t\t\t => Get Spectro Data')
else:
	print('\t\t\t => Read Raw Data')
	if regrid : print('\t\t\t => regridding : ',regridding)
	print('\t\t\t => Fourier Transform')
	if vis: print('\t\t\t => Visualization')
	if FISTA_CSV : print('\t\t\t => Export FISTA readable CSV')
	if Save: print('\t\t\t => Saving Output Images !')
	else: 
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 12)
		print('\t\t\t => Reconstructed Images WON\'T be saved !')
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)

# if not TPI : CPLX, parameters = ExtractDataFromRawData(source_file,verbose,vis,HeaderOnly)
if not TPI and not Spectro: CPLX, parameters = ReadFastSiemensRAD(source_file,HeaderOnly)
# if TPI : CPLX, parameters,KX,KY,KZ = ExtractDataFromRawData_TPI(source_file,verbose,vis,HeaderOnly)
if TPI : 

	if MIA :
		CPLX, parameters,KX,KY,KZ = ReadFastSiemensTPI_MIA_MonoEcho(source_file,HeaderOnly,NS_TPI)
	else :	
		CPLX, parameters,KX,KY,KZ = ReadFastSiemensTPI_MultiEcho(source_file,HeaderOnly,NS_TPI)
	# Sum of averages if performed now to be correctly writen for FISTA file
	# CPLX=numpy.sum(CPLX[:,:,:,:],0);
	if CPLX.shape[0] != 1:
		CPLX=numpy.sum(CPLX[:,:,:,:,:],0)
	elif CPLX.shape[0] == 1 and CPLX.shape[3] != 1:
		CPLX = numpy.squeeze(CPLX)
	elif CPLX.shape[0] == 1 and CPLX.shape[3] == 1:
		CPLX = numpy.reshape(CPLX,(CPLX.shape[1],CPLX.shape[2],CPLX.shape[3],CPLX.shape[4]))	
	print(KX.shape)
	print(CPLX.shape)
	print("----------------------------------")
	# if str(parameters[7]) == '23Na': CPLX = CPLX[1:,:,:]
	# if not os.path.isfile("TPI_Kspace_positions_values_31P_bouleFawziTubes_p075_40960proj_FA15_TE4_5.csv") :
	if FISTA_CSV:
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 15)
		print('INFO    : Writing CSV Files for FISTA')
		print(CPLX.shape)
		#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
		for echo in range(parameters[24]):
			FISTApath = os.path.dirname(OutputPath)
			FISTApath = os.path.join( FISTApath, 'FISTA_KspaceValues_Echo%s.csv' %echo )
			print('Hello, this is FISTApath,',FISTApath[0])
			if not os.path.isfile(FISTApath) :
				# f=open("TPI_Kspace_positions_values_31P_bouleFawziTubes_p075_40960proj_FA10_TE4_5.csv","w")
				print('INFO    : Writing CSV file for echo ', echo+1)
				f=open(FISTApath,"w")
				#####f.write(float(KX))
				### en changeant l'incrément de la boucle on retourne plus ou moins de points de la trajectoire ! faire attention !!
				# for i in range(0,KX.shape[0],100):
				# echo = 0;
				print('Hello, this is CPLX shape ,',CPLX.shape[0])  
				print('Hello, this is KX shape 1,',KX.shape[0])
				print('Hello, this is KX shape 2,',KX.shape[1])  
				print('Hello, this is an example of KX[0,0]',KX[0,0])  
				print('Hello, this is an example of KY[0,0]',KY[0,0])
				print('Hello, this is an example of KZ[0,0]',KZ[0,0])   
				print('Hello, this is an example of numpy CPLX',CPLX[0,0,0,0])
				print(KX.shape)
				print(CPLX.shape)				
				for coil in range(CPLX.shape[0]):
				# for coil in range(1):
					for i in range(0,KX.shape[0]):
						# for j in range(0,KX.shape[1],10):
						for j in range(0,KX.shape[1]):
							f.write(str(coil))
							f.write(',')
							f.write(str(i))
							f.write(',')
							f.write(str(j))
							f.write(',')
							f.write(str(float(KX[i,j])))
							f.write(',')
							f.write(str(float(KY[i,j])))
							f.write(',')
							f.write(str(float(KZ[i,j])))
							f.write(',')
							f.write(str(numpy.real(CPLX[coil,i,echo,j])))
							f.write(',')
							f.write(str(numpy.imag(CPLX[coil,i,echo,j])))
							f.write("\n")
				f.close()
		
	# from scipy.spatial import Voronoi
	# from scipy.spatial import Delaunay
	# from processingFunctions import tetrahedron_volume
	# points = numpy.column_stack((numpy.ravel(KX[0:KX.shape[0]:100,0:KX.shape[1]:10]),numpy.ravel(KY[0:KY.shape[0]:100,0:KY.shape[1]:10]),numpy.ravel(KZ[0:KZ.shape[0]:100,0:KZ.shape[1]:10])))
	# points = numpy.column_stack((numpy.ravel(KX[5:5000:5,:]),numpy.ravel(KY[5:5000:5,:]),numpy.ravel(KZ[5:5000:5,:])))

	# #####compute Voronoi tesselation
	# vor = Voronoi(points)
	# print len(vor.regions)
	# ####print vor.vertices
	# from scipy.spatial import ConvexHull
	# volume = numpy.zeros(len(vor.regions))
	# for node in range(len(vor.regions)):
		# if node <(points.shape[0]) :
			# print node
			# print vor.regions[node]
			# if vor.regions[node] != []:
				# xv = vor.points[numpy.mod(vor.regions[node],points.shape[0]),0]
				# yv = vor.points[numpy.mod(vor.regions[node],points.shape[0]),1]
				# zv = vor.points[numpy.mod(vor.regions[node],points.shape[0]),2]
				# xv = vor.points[vor.regions[node],0]
				# yv = vor.points[vor.regions[node],1]
				# zv = vor.points[vor.regions[node],2]
				# pts=numpy.zeros(shape=(len(xv),3))
				# for i in range(len(xv)):
					# pts[i]=[xv[i],yv[i],zv[i]]
				# print (xv.shape,yv.shape, zv.shape)
				# if node >2900: print (xv,yv, zv)
				
				# if ((xv !=[] and yv !=[] and zv != []) and (len(xv)>=11)):
				# if ((xv !=[] and yv !=[] and zv != []) and (len(xv)>3)):
					# VoronoiCellConvexHull = ConvexHull(pts,qhull_options="FA")
					# print VoronoiCellConvexHull.points
					# dt = Delaunay(VoronoiCellConvexHull.points)
					# tetras = dt.points[dt.simplices]
					# print tetras
					# volume[node] = numpy.sum(tetrahedron_volume(tetras[:, 0], tetras[:, 1], tetras[:, 2], tetras[:, 3]))
					# print VoronoiCellConvexHull.vertices
					# volume[node] = VoronoiCellConvexHull.volume()
					# print 'Volume',node,' == ',volume[node]/200**3
				# print tetras.shape	
					
	# #weightVoronoi=numpy.unique(numpy.round(area,9))
	# #weightVoronoi=numpy.sort(weightVoronoi)
	# del(points);del(vor);del(xv);del(yv);del(zv);
	# if not os.path.isfile("TPI_Kspace_coefs.csv") :
		# f=open("TPI_Kspace_coefs.csv","w")
		# for i in range(len(volume)):
			# f.write(str(float(numpy.round(volume[i],9))))
			# f.write("\n")
		# f.close()
		
#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 10)
if HeaderOnly : print('Header Information Extracted')
else : 
	if not Spectro : print('Data and parameters extracted')
#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)

# KaiserBesselRegridding(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],CPLX)

if Spectro:
	Spectrum = ReadSiemensSpectro(source_file,verbose)

if not HeaderOnly:
	if regridding and rad :
		if parameters[15] and parameters[16] and not args.ManualBW:
			# ReconstructedImg=Radial_Regridding(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],False,CPLX,Interpolation,verbose,parameters[15],parameters[16])
			ReconstructedImg,Phase=KaiserBesselRegridding(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],False,CPLX,Interpolation,verbose,parameters[15],parameters[16])
		else :
			# ReconstructedImg=Radial_Regridding(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],False,CPLX,Interpolation,verbose)
			ReconstructedImg,Phase=KaiserBesselRegridding(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],False,CPLX,Interpolation,verbose)
	if TPI : 
		# ReconstructedImg=Direct_Reconstruction_3D(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],False,CPLX)
		# ReconstructedImg=TPI_Regridding(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,Interpolation,verbose)
		if NS_TPI:
			if MIA : parameters[24]=3
			# ReconstructedImg=KaiserBesselTPI(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity)
			# if SaveKspace : ReconstructedImg, Phase, Kspace =KaiserBesselTPI_ME(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity,int(parameters[24]),SaveKspace,UseFullDCF)
			if SaveKspace : ReconstructedImg, Phase, Kspace =KaiserBesselTPI_ME(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity,int(parameters[24]),SaveKspace)
			# if SavePhase and not SaveKspace : ReconstructedImg, Phase =KaiserBesselTPI_ME(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity,int(parameters[24]),SaveKspace,UseFullDCF)
			if SavePhase and not SaveKspace : ReconstructedImg, Phase =KaiserBesselTPI_ME(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity,int(parameters[24]),SaveKspace)
			else : 
				# ReconstructedImg, Phase =KaiserBesselTPI_ME(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity,int(parameters[24]),SaveKspace,UseFullDCF)
				ReconstructedImg, Phase, Abs_Sum_of_Regridded_kspace =KaiserBesselTPI_ME(parameters[18],parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity,int(parameters[24]),SaveKspace)
				del(Phase)
		else:
			ReconstructedImg=KaiserBesselTPI(20000,parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,parameters[20],parameters[21],parameters[8],verbose,PSF_ND,PSF_D,B1sensitivity)
		# ReconstructedImg=KaiserBesselTPI(4498,parameters[1],parameters[2],parameters[3],parameters[5],parameters[6],parameters[7],False,CPLX,KX,KY,KZ,verbose)
	if not TPI and not regrid and Cartesian and not Cart3D and not Spectro:
		ReconstructedImg=Direct_Reconstruction_2D(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],False,CPLX)
	if not TPI and not regrid and Cart3D:
		ReconstructedImg=Direct_Reconstruction_3D(parameters[0],parameters[1],parameters[2],parameters[3],parameters[19],parameters[5],parameters[6],parameters[7],False,CPLX)
	
	# f = open("matrix.txt", "w")
	# f.write(ReconstructedImg)
	# f.close()
	
	# if SavePhase:
		# if regrid and regridding.upper() == 'RAD' :
			# pix_x=float(float(parameters[8])/(float(parameters[1])*float(parameters[5])*2)) #(= FOV_x / (nbpts (*oversamplingFactor)*2))
			# pix_y=float(float(parameters[9])/(float(parameters[1])*float(parameters[5])*2)) #(= FOV_y / (nbpts (*oversamplingFactor)*2) (On a deux radiales pour une dim de FOV)
		
		# from nifty_funclib import SaveArrayAsNIfTI
		# SaveArrayAsNIfTI(Phase,pix_x,pix_y,float(parameters[10]),"Phase.nii")
		# print "Phase image saved in current directory as Phase.nii"
	if Save:	
		
		# Get pixel dimensions : 
		if regrid and regridding.upper() == 'RAD' :
			pix_x=float(float(parameters[8])/(float(parameters[1])*float(parameters[5])*2)) #(= FOV_x / (nbpts (*oversamplingFactor)*2))
			pix_y=float(float(parameters[9])/(float(parameters[1])*float(parameters[5])*2)) #(= FOV_y / (nbpts (*oversamplingFactor)*2) (On a deux radiales pour une dim de FOV)
		else:
			pix_x=float(2*float(parameters[8])/(float(parameters[1])*float(parameters[5])))
			pix_y=float(float(parameters[9])/(float(parameters[0]))) #(= FOV_y / nbLines (EN CARTESIEN) (NO oversampling in this direction))
		
		if TPI : 
			pix_x=float(parameters[21])
			pix_y=float(parameters[21])
			pix_z=float(parameters[21])
		
		from nifty_funclib import SaveArrayAsNIfTI,SaveArrayAsNIfTI_2
		if not TPI : 
			SaveArrayAsNIfTI(ReconstructedImg,pix_x,pix_y,float(parameters[10]),OutputPath)
			# SaveArrayAsNIfTI_2(ReconstructedImg,pix_x,pix_y,float(parameters[10]),NbPoints,NbLines,NbSlices,rad,orientation,OutputPath)
	
			# SaveArrayAsNIfTI_2(ReconstructedImg,pix_x,pix_y,float(parameters[10]),int((parameters[1])*(parameters[5])*2),int(parameters[0]),int(parameters[4]),rad,str(parameters[12]),OutputPath)
		if TPI :
			if MIA : parameters[24]=3
			if int(parameters[24])==1:
				Hpath, Fname = os.path.split(str(OutputPath))
				Fname = Fname[0].split('.')
				if SavePhase :
					OutputPath = os.path.join( Hpath + '/' + Fname[0] + '_KBgrid_MODULE.nii')
					SaveArrayAsNIfTI(ReconstructedImg[0,:,:,:],pix_x,pix_y,pix_z,OutputPath)
					OutputPath = os.path.join( Hpath + '/' + Fname[0] + '_KBgrid_PHASE.nii')
					SaveArrayAsNIfTI(Phase[0,:,:,:],pix_x,pix_y,pix_z,OutputPath)
				else :
					OutputPath = os.path.join( Hpath + '/' + Fname[0] + '_KBgrid_MODULE.nii')
					SaveArrayAsNIfTI(ReconstructedImg[0,:,:,:],pix_x,pix_y,pix_z,OutputPath)
					
				if SaveKspace :
					OutputPath = os.path.join( Hpath + '/' + Fname[0] + '_KB_GriddedKspace.nii')
					SaveArrayAsNIfTI(Kspace[0,0,:,:,:],pix_x,pix_y,pix_z,OutputPath)

			if int(parameters[24])>1:
				print("INFO    : Saving Multiple Images")
				Hpath, Fname = os.path.split(str(OutputPath))
				Fname = Fname.split('.')
				for echo in range(int(parameters[24])):
					if SavePhase :
						OutputPath = os.path.join( Hpath + '/' + Fname[0] + "_KBgrid_MODULE_Echo%s.nii" %echo )
						SaveArrayAsNIfTI(ReconstructedImg[echo,:,:,:],pix_x,pix_y,pix_z,OutputPath)
						OutputPath = os.path.join( Hpath + '/' + Fname[0] + "_KBgrid_PHASE_Echo%s.nii" %echo )
						SaveArrayAsNIfTI(Phase[echo,:,:,:],pix_x,pix_y,pix_z,OutputPath)
					else:
						OutputPath = os.path.join( Hpath + '/' + Fname[0] + "_KBgrid_MODULE_Echo%s.nii" %echo )
						SaveArrayAsNIfTI(ReconstructedImg[echo,:,:,:],pix_x,pix_y,pix_z,OutputPath)
				if SumEchoes:
					OutputPath = os.path.join( Hpath + '/' + Fname[0] + "_KBgrid_SumCplx_Echoes.nii")
					SaveArrayAsNIfTI(ReconstructedImg[echo,:,:,:],pix_x,pix_y,pix_z,OutputPath)
					# if SaveKspace :
						# Hpath, Fname = os.path.split(OutputPath)
						# Fname = Fname.split('.')
						# OutputPath = os.path.join( str(Hpath), "CPLX_Kspace_Echo%s.nii" %echo )
						# SaveArrayAsNIfTI(Kspace[echo,,:,:,:],pix_x,pix_y,pix_z,OutputPath)
	else:
		if not Spectro:
			#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 12)
			print('WARNING : Image not saved')
			print('ERROR   : WRONG EXTENSION (should be .nii)')
			#windll.Kernel32.SetConsoleTextAttribute(std_output_hdl, 7)
		
end = time.time()
print("INFO    : Computation time =", end - start)
