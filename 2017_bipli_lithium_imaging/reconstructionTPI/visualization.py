# -*- coding:Utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, ginput, show, axis
# import pylab
# Visualization library

def PlotImg(matrix):
	plt.imshow(matrix, cmap='gray', interpolation='nearest')
	plt.clim(0, 0.0001)
	plt.show()
	
def PlotImg2(matrix):
	plt.imshow(matrix, cmap='gray', interpolation='nearest')
	plt.clim(0, 255)
	plt.show()
	
def PlotImgMag(matrix):
	plt.imshow(matrix, cmap='gray', interpolation='nearest')
	plt.clim(np.amin(matrix), np.amax(matrix)/1200)
	plt.show()
	
def PlotImgMag2(matrix):
	plt.imshow(matrix, cmap='gray', interpolation='nearest')
	plt.clim(np.amin(matrix), np.amax(matrix)/30)
	plt.show()
	
def PlotReconstructedImage(matrix):
	plt.imshow(matrix, cmap='gray', interpolation='nearest')
	plt.clim(np.amin(matrix), np.amax(matrix))
	plt.show()
	
def DefineROIonImage(matrix):
	plt.imshow(matrix, cmap='gray', interpolation='nearest')
	plt.clim(np.amin(matrix), np.amax(matrix))
	plt.show()
	while (button==1):
		pts = ginput(0, timeout=10, show_clicks=True,mouse_add=1, mouse_pop=3, mouse_stop=2) # it will wait for both button click
	pts=array(pts)
	x=pts[:,0]
	y=pts[:,1]
	plot(x,y,'-o')
	plt.show()
	
	# List of possible color maps :	
	# cmaps = [('Sequential',     ['Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
								  # 'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         # ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool', 'copper', 'gist_heat', 'gray', 'hot', 'pink', 'spring', 'summer', 'winter']),
         # ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral', 'seismic']),
         # ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3']),
         # ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern', 'brg', 'CMRmap', 'cubehelix', 'gnuplot', 'gnuplot2', 'gist_ncar',
                              # 'nipy_spectral', 'jet', 'rainbow', 'gist_rainbow', 'hsv', 'flag', 'prism'])]
							  
def PlotSpectrum(Spectrum):

	
	t = np.arange(0.0, (len(Spectrum)), 1)
	s = np.real((Spectrum[0:len(Spectrum)]))
	plt.subplot(241)
	plt.xlabel('point')
	plt.ylabel('Amplitude')
	plt.title('Real')
	plt.plot(t, s)
	
	s = np.imag((Spectrum[0:len(Spectrum)]))
	plt.subplot(245)
	plt.title('Imag')
	plt.xlabel('point')
	plt.ylabel('Amplitude')
	plt.plot(t, s)
	
	s = np.absolute((Spectrum[0:len(Spectrum)]))
	plt.subplot(242)
	plt.title('Modulus')
	plt.xlabel('point')
	plt.ylabel('Amplitude')
	plt.plot(t, s)
	
	s = np.angle((Spectrum[0:len(Spectrum)]))
	plt.subplot(246)
	plt.title('Phase')
	plt.xlabel('point')
	plt.ylabel('Amplitude')
	plt.plot(t, s)
	
	s = np.absolute(np.fft.fft((Spectrum[0:len(Spectrum)])))
	plt.subplot(243)
	plt.title('FFT Modulus')
	plt.xlabel('point')
	plt.ylabel('Amplitude')
	plt.plot(t, s)
	
	s = np.angle(np.fft.fft((Spectrum[0:len(Spectrum)])))
	plt.subplot(247)
	plt.title('FFT Phase')
	plt.xlabel('point')
	plt.ylabel('Amplitude')
	plt.plot(t, s)
	
	s = np.real(np.fft.fft((Spectrum[0:len(Spectrum)])))
	plt.subplot(244)
	plt.title('FFT real')
	plt.xlabel('point')
	plt.ylabel('Amplitude')
	plt.plot(t, s)
	
	s = np.imag(np.fft.fft((Spectrum[0:len(Spectrum)])))
	plt.subplot(248)
	plt.title('FFT imag')
	plt.xlabel('point')
	plt.ylabel('Amplitude')
	plt.plot(t, s)
	
	
	# plt.grid(True)
	# plt.savefig("test.png")
	plt.show()