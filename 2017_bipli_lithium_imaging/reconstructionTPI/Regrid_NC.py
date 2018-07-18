import os
import argparse
import numpy
import math,scipy.ndimage
from scipy.interpolate import griddata
from visualization import PlotImg,PlotImgMag,PlotReconstructedImage,PlotImg2,PlotImgMag2,DefineROIonImage

def CalculateKaiserBesselKernel(width, overgriddingfactor, length):	
	
	if not length:
		length=32
	if length <2:
		length = 2
	a=overgriddingfactor
	w=width
	
	beta=numpy.pi*numpy.sqrt((w**2/a**2)*((a-0.5)**2)-0.8)		#Rapid Gridding Reconstruction With a Minimal Oversampling Ratio P. J. Beatty et al, 2005
	u=numpy.arange(0,length-1,dtype=float)/(length-1) * (int(w)/2)
	kernel=beta*numpy.sqrt(1-((2*u[numpy.absolute(u) < int(w)/2])/w)**2) 	#Selection of a Convolution Function for Fourier Inversion using Griding, Jackson et al 1991
	KBkernel=(numpy.i0(kernel))/width
	# KBkernel=KBkernel/KBkernel[0]
	print (KBkernel)
	
	return KBkernel, u,beta
	
def simple_poly_area(x, y):
    # For short arrays (less than about 100 elements) it seems that the
    # Python sum is faster than the numpy sum. Likewise for the Python
    # built-in abs.
    return .5 * abs(sum(x[:-1] * y[1:] - x[1:] * y[:-1]) +
                    x[-1] * y[0] - x[0] * y[-1])

def KaiserBesselRegridding():

	i=0
	Kpos=open("C:\\Users\AC243636\Documents\Versioned_code\python\Data_Carole\samples.csv",'r')
	Kdata=open("C:\\Users\AC243636\Documents\Versioned_code\python\Data_Carole\datavalues_bab_1000NEX.csv",'r')
	KPOS = numpy.zeros(shape=(129408,2), dtype=numpy.float32)	
	KVAL = numpy.zeros(shape=(129408,1), dtype=numpy.complex64)			
	for line in Kpos:
		values=line.split(',')
		KPOS[i,0] = values[0]
		KPOS[i,1] = values[1]
		i=i+1
		if i==6066:
			break
	i=0
	for line in Kdata:
		if i%4==0:
			values=line.split(',')
			Re = values[0]
			Imag = values[1]
			KVAL[i] = complex(float(Re),float(Imag))
			i=i+1
		if i==6066:
			break

	#Compute Kaiser Bessel
	NormalizedKernel, u, beta = CalculateKaiserBesselKernel(3,2,4)
	NormalizedKernelflip=numpy.flipud(NormalizedKernel)
	# print((NormalizedKernelflip))
	NormalizedKernelflip=numpy.delete(NormalizedKernelflip,2)
	NormalizedKernel=numpy.append(NormalizedKernelflip,NormalizedKernel)
	print((NormalizedKernel))
	NormalizedKernel=numpy.mat(NormalizedKernel)
	NormalizedKernel2D=numpy.transpose(NormalizedKernel)*(NormalizedKernel)	
	print(NormalizedKernel2D)
	print((NormalizedKernel2D.shape))
	
	
	#Perform Gridding
	
	#1) Generate K space trajectory

	from scipy.spatial import Voronoi, voronoi_plot_2d
	import matplotlib.pyplot as plt

	# points = numpy.column_stack((a,b))
	points = numpy.column_stack((numpy.ravel(KPOS[:,0]),numpy.ravel(KPOS[:,1])))

	# compute Voronoi tesselation
	vor = Voronoi(points)
	voronoi_plot_2d(vor)
	plt.show()
	###########print (len(vor.regions))
	area = numpy.zeros(len(vor.regions))
	for node in range(len(vor.regions)):
		xv = vor.vertices[vor.regions[node],0]
		yv = vor.vertices[vor.regions[node],1]
		#############print (xv,yv)
		if ((xv !=[] and yv !=[]) and (len(xv)==4)):
			area[node] = simple_poly_area(xv, yv)
			#############print ("Aire ",node," == ",simple_poly_area(xv, yv))
	
	# weightVoronoi=numpy.unique(numpy.round(area,9))
	# weightVoronoi=numpy.sort(numpy.round(area,9))
	# print weightVoronoi.shape
	weightVoronoi=numpy.round(area,9)+(1/(int(2022)))
	weightVoronoi=weightVoronoi/max(weightVoronoi)
	weightVoronoi=numpy.append(weightVoronoi,1)
	print(weightVoronoi.shape)
	weightVoronoi[0]=1e-7 
	f=open("VornoiCoefCarole.txt","w")
	for j in range(0,6065):
		f.write(str(weightVoronoi[j]))
		f.write("\n")
	f.close()
	del(points);del(vor);del(area);del(xv);del(yv);
	return
	# LinearWeigths=numpy.linspace(1/(2022), 1,2022)
	# LinearWeigths=numpy.delete(LinearWeigths,0)
	# LinearWeigths=numpy.append(LinearWeigths,1)
	# print LinearWeigths.shape
	
	imsize = 2048

	size = 2048*1.5
	
	# On utilise une grille 2 fois plus grande pour minimiser les erreurs de regridding et on crop apres
	# size = NbPoints*OverSamplingFactor*2*2

	# Regridded_kspace = numpy.zeros(shape=(int(NbSlice),int(NbCoils),int(size)+2,int(size)+2), dtype=numpy.complex64)		
	# Coil_Combined_Kspace = numpy.zeros(shape=(int(NbSlice),int(size)+2,int(size)+2))
	Regridded_kspace = numpy.zeros(shape=(int(size),int(size)), dtype=numpy.complex64)		
	ImageMag =  numpy.zeros(shape=(int(size),int(size)), dtype=numpy.float32)	
	# We generate a random value (Uniform Distribution (Gaussian ?)) and compare it with some threshold to remove the line
	Val=0j
	for m in range(129408):

		x_current=numpy.round(KPOS[m][0]/numpy.amax(KPOS[:,0])*size/2)
		y_current=numpy.round(KPOS[m][1]/numpy.amax(KPOS[:,1])*size/2)
		# Val=DataAvg[j][i][l][m]
		# Val=DataAvg[m]*weightVoronoi[m]
		Val=KVAL[m,0]*LinearWeigths[m%64]
		# print(Val)
		
		for a in range(-1,1,1):
			for b in range(-1,1,1):
				# print a,b
				# print Val*NormalizedKernel2D[a+1,b+1]
				# print type(Val)
				# print type(KVAL[m,0])
				# print type(Regridded_kspace[y_current+a][x_current+b])

				Regridded_kspace[y_current+a][x_current+b]=Regridded_kspace[y_current+a][x_current+b]+Val*NormalizedKernel2D[a+1,b+1]
				# print Regridded_kspace[y_current+a][x_current+b]
	
	ImageMag[:][:]=(numpy.fft.fftshift(numpy.fft.ifftn(numpy.fft.fftshift((Regridded_kspace[:][:])))))	
	PlotReconstructedImage(ImageMag)

	# PlotReconstructedImage((Coil_Combined_Kspace_Module[j,imsize/2:imsize+imsize/2,imsize/2:imsize+imsize/2])/C)
	
	PlotImgMag(numpy.absolute((ImageMag)))
	
	print("[DONE]")
	from nifty_funclib import SaveArrayAsNIfTI
	SaveArrayAsNIfTI(ImageMag,1,1,1,"Carole.nii")
	
	
	return ImageMag	