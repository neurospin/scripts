# -*- coding:Utf-8 -*-
#
# Author : Arthur Coste
# Date : February 2015
# Purpose : Simple Image Processing Operations
#             
#--------------------------------------------------------------------------------------------------------------------------------- 
import numpy as np

def ThresholdImage(Img,value):

    size = Img.shape
    binary_mask=np.zeros(shape=size)
    
    for i in range(len(Img)):
        for j in range (len(Img[0])):
            if len(size)==3: 
                for k in range(len(Img[1])):
                    if Img[i,j,k]>=value:
                        binary_mask[i,j,k]=1
                    else:
                        binary_mask[i,j,k]=0
            else :
                if Img[i,j]>=value:
                    binary_mask[i,j]=1
                else:
                    binary_mask[i,j]=0
    return binary_mask
            
def MaskImage(Img,Mask):
    
    if Img.shape != Mask.shape:
        print('ERROR : Mask and Images of different shape !')
        # raise ERROR
    else:
        MaskedImage=np.zeros(Img.shape)
        
        for i in range(len(Img)):
            for j in range (len(Img[0])):
                if len(Img.shape)==3: 
                    for k in range(len(Img[1])):
                        if Mask[i,j,k]!=0:
                            MaskedImage[i,j,k]=Img[i,j,k]
                        else:
                            MaskedImage[i,j,k]=0
                else :
                    if Mask[i,j]!=0:
                        MaskedImage[i,j]=Img[i,j]
                    else:
                        MaskedImage[i,j]=0
        return MaskedImage
    
def MeanImage(Img):
        
    mean=0
    for i in range(len(Img)):
            for j in range (len(Img[0])):
                if len(Img.shape)==3: 
                    for k in range(len(Img[1])):
                        mean = mean+Img[i,j,k]
                else :
                    mean = mean+Img[i,j]
    mean=mean/Img.size                
    return mean
    
def MinMaxImage(Img):

    Min=1e999;Max=0
    for i in range(len(Img)):
        for j in range (len(Img[0])):
            if len(Img.shape)==3: 
                for k in range(len(Img[1])):
                    if Img[i,j,k] > Max :
                        Max=Img[i,j,k]
                    if Img[i,j,k] < Min :
                        Min=Img[i,j,k]
            else :
                if Img[i,j] > Max :
                    Max=Img[i,j]
                if Img[i,j] < Min :
                    Min=Img[i,j]
    return Min,Max    
    
def GetSquareROI(Img,size,y_upcorner,x_upcorner):

    ROI = np.zeros(shape=(size,size))
    for i in range(size):
        for j in range (size):
            ROI[i,j]=Img[int(x_upcorner+i)][int(y_upcorner+j)]
            
    return ROI
        
    
def AddImages(Img1,Img2):

    if (Img1.shape != Img2.shape):
        if Img.shape != Mask.shape:
            print('ERROR : Images of different size !')
    else:
        size = Img1.shape
        sum_img=np.zeros(shape=size)

        for i in range(len(Img)):
            for j in range (len(Img[0])):
                if len(Img.shape)==3: 
                    for k in range(len(Img[1])):
                        sum_img[i,j,k] =  Img1[i,j,k]+Img2[i,j,k]
                else :
                    sum_img[i,j] =  Img1[i,j]+Img2[i,j]
        return sum_img
        
def SubtractImages(Img1,Img2,abs):

    if (Img1.shape != Img2.shape):
        if Img.shape != Mask.shape:
            print('ERROR : Images of different size !')
    else:
        size = Img1.shape
        diff_img=np.zeros(shape=size)

        for i in range(len(Img)):
            for j in range (len(Img[0])):
                if len(Img.shape)==3: 
                    for k in range(len(Img[1])):
                        if abs : diff_img[i,j,k] = np.absolute(Img1[i,j,k]-Img2[i,j,k])
                        else : diff_img[i,j,k] =  Img1[i,j,k]-Img2[i,j,k]
                else :
                    if abs : diff_img[i,j] =  np.absolute(Img1[i,j]-Img2[i,j])
                    else : diff_img[i,j] =  Img1[i,j]-Img2[i,j]
        return diff_img

def ComputeFullImageSNR(Image):
    if Image ==[]:
        print('ERROR : Empty Image !')
    else :
        import numpy
        meanImage=numpy.mean(Image)
        stdev=numpy.std(Image)
        GlobalSNR = 20*numpy.log(meanImage/stdev)
        return GlobalSNR    

def ComputeDifferenceMaps(Image1,Image2,signed):
        if Image1 == [] or Image2 == [] :
            print('ERROR : Empty Image !')
        if (Image1.shape != Image2.shape):
            print('ERROR : Images of different size !')    
        else :
            if signed == False:
                UnsignedDiffImage=numpy.absolute(Image1-Image2)
                return UnsignedDiffImage
            else :    
                SignedDiffImage=(Image1-Image2)
                return SignedDiffImage
                
def NRMSE(Image1,Image2):
    import nibabel as nib
    import numpy
    Image1 = nib.load(Image1)
    Image2 = nib.load(Image2)
    Image1 =Image1.get_data()
    Image2 =Image2.get_data()
    Image1 =numpy.squeeze(Image1)
    Image2 =numpy.squeeze(Image2)
    
    RMSE = numpy.zeros(shape=Image1.shape, dtype=numpy.float32)
    RMSE = numpy.sqrt(((Image1 - Image2) ** 2) / numpy.sum(Image1**2))
    # NRMSE= RMSE / (numpy.amax(Image2)- numpy.amin(Image2))
    # NRMSE= RMSE / (numpy.sum(Image1))
    NRMSE= RMSE / (numpy.amax(Image1))
    NRMSEval = numpy.sum(NRMSE)/NRMSE.size
    # print (RMSE)
    # print( NRMSE)
    from visualization import PlotReconstructedImage
    PlotReconstructedImage(NRMSE)
    from nifty_funclib import SaveArrayAsNIfTI
    SaveArrayAsNIfTI(NRMSE,1,1,1,'NRMSE.nii')    
    return NRMSEval
    
# def compute_ssim(Image1, Image2,mask):
def compute_ssim(Image1, Image2):
# def compute_ssim():
    # http://isit.u-clermont1.fr/~anvacava/codes/ssim.py
    import numpy
    import scipy.ndimage
    # import scipy.misc
    from numpy.ma.core import exp
    from scipy.constants.constants import pi
    import nibabel as nib
    Image1 = nib.load(Image1)
    Image2 = nib.load(Image2)
    # mask = nib.load(mask)
    img_mat_1 =Image1.get_data()
    img_mat_2 =Image2.get_data()
    # mask =mask.get_data()
    img_mat_1 =numpy.squeeze(img_mat_1)
    img_mat_2 =numpy.squeeze(img_mat_2)
    # mask =numpy.squeeze(mask)
    from visualization import PlotReconstructedImage
    # PlotReconstructedImage(img_mat_1)
    # PlotReconstructedImage(img_mat_2)
    
    img_mat_1 =img_mat_1/numpy.amax(img_mat_1)
    img_mat_2 =img_mat_2/numpy.amax(img_mat_2)
    
    # img_mat_1 =  scipy.misc.imread(Image1)
    # img_mat_2 =  scipy.misc.imread(Image2)
    # img_mat_1 = numpy.sum(img_mat_1[:,:,:],2)
    # img_mat_2 = numpy.sum(img_mat_2[:,:,:],2)
    # for i in range(img_mat_1.shape[0]):
        # for j in range(img_mat_1.shape[1]):
            # if mask[i,j]==0:
                # img_mat_1[i,j]=0
                # img_mat_2[i,j]=0
    
    # img_mat_1 =numpy.reshape(img_mat_1,img_mat_2.shape)
    #Variables for Gaussian kernel definition
    gaussian_kernel_sigma=1.5
    gaussian_kernel_width=11
    gaussian_kernel=numpy.zeros((gaussian_kernel_width,gaussian_kernel_width))
    
    #Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i,j]=\
            (1/(2*pi*(gaussian_kernel_sigma**2)))*\
            exp(-(((i-5)**2)+((j-5)**2))/(2*(gaussian_kernel_sigma**2)))

    #Convert image matrices to double precision (like in the Matlab version)
    img_mat_1=img_mat_1.astype(numpy.float)
    img_mat_2=img_mat_2.astype(numpy.float)
    
    #Squares of input matrices
    img_mat_1_sq=img_mat_1**2
    img_mat_2_sq=img_mat_2**2
    img_mat_12=img_mat_1*img_mat_2
    
    #Means obtained by Gaussian filtering of inputs
    print(gaussian_kernel.shape)
    print(img_mat_1.shape)
    img_mat_mu_1=scipy.ndimage.filters.convolve(img_mat_1,gaussian_kernel)
    img_mat_mu_2=scipy.ndimage.filters.convolve(img_mat_2,gaussian_kernel)
        
    #Squares of means
    img_mat_mu_1_sq=img_mat_mu_1**2
    img_mat_mu_2_sq=img_mat_mu_2**2
    img_mat_mu_12=img_mat_mu_1*img_mat_mu_2
    
    #Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq=scipy.ndimage.filters.convolve(img_mat_1_sq,gaussian_kernel)
    img_mat_sigma_2_sq=scipy.ndimage.filters.convolve(img_mat_2_sq,gaussian_kernel)
    
    #Covariance
    img_mat_sigma_12=scipy.ndimage.filters.convolve(img_mat_12,gaussian_kernel)

    #Centered squares of variances
    img_mat_sigma_1_sq=img_mat_sigma_1_sq-img_mat_mu_1_sq
    img_mat_sigma_2_sq=img_mat_sigma_2_sq-img_mat_mu_2_sq
    img_mat_sigma_12=img_mat_sigma_12-img_mat_mu_12;

    #c1/c2 constants
    #First use: manual fitting
    # c_1=6.5025
    # c_2=58.5225

    #Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    # l=(2^32)-1
    # l=numpy.amax(img_mat_1)
    l=1
    k_1=0.01
    c_1=(k_1*l)**2
    k_2=0.03
    c_2=(k_2*l)**2

    #Numerator of SSIM
    num_ssim=(2*img_mat_mu_12+c_1)*(2*img_mat_sigma_12+c_2)
    #Denominator of SSIM
    den_ssim=(img_mat_mu_1_sq+img_mat_mu_2_sq+c_1)*\
    (img_mat_sigma_1_sq+img_mat_sigma_2_sq+c_2)
    #SSIM
    ssim_map=num_ssim/den_ssim
    index=numpy.average(ssim_map)
    from visualization import PlotReconstructedImage
    # PlotReconstructedImage(ssim_map)
    from nifty_funclib import SaveArrayAsNIfTI
    SaveArrayAsNIfTI(ssim_map,1,1,1,'ssim_map.nii')    

    return index    
    
def DivideImages(Img1,Img2):

    if (Img1.shape != Img2.shape):
        if Img.shape != Mask.shape:
            print('ERROR : Images of different size !')
    else:
        # size = Img1.shape
        # div_img=np.zeros(shape=size)
        div_img=Img1/Img2
        return div_img

# def Compare2Distributions(Img1,Img2):

    # if (Img1.shape != Img2.shape):
        # if Img.shape != Mask.shape:
            # print 'ERROR : Images of different size !'
    
    # else:
def CompareB1maps(Image1,Image2):

    import nibabel as nib
    Image1 = nib.load(Image1)
    Image2 = nib.load(Image2)
    img_mat_1 =Image1.get_data()
    img_mat_2 =Image2.get_data()
    
    if (img_mat_1.shape != img_mat_2.shape):
        print('ERROR : Images of different size !')
            
    else:
        import numpy
        import matplotlib.pyplot as plt
        from scipy import stats
        x=numpy.reshape(img_mat_1, (1, img_mat_1.size))
        y=numpy.reshape(img_mat_2, (1, img_mat_2.size))
        mask1=numpy.where(x == 0)
        mask2=numpy.where(y == 0)
        x[mask2]=0
        y[mask1]=0
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        print(slope, intercept, r_value, p_value, std_err)
        print("r-squared:", r_value**2)
        plt.scatter(x/10, y/10, alpha=1)
        plt.plot(numpy.arange(0,numpy.amax(img_mat_1)/10), numpy.arange(0,numpy.amax(img_mat_1)/10)*slope + intercept, 'k')
        plt.plot(numpy.arange(0,numpy.amax(img_mat_1)/10), numpy.arange(0,numpy.amax(img_mat_1)/10)*1 + 0, 'r')
        plt.xlabel('31P-BAFI')
        plt.ylabel('31P-XFL')
        plt.title('Flip Angle')
        plt.text(10, 65, 'slope=%s'%(slope) )
        plt.axis([0, (numpy.amax(img_mat_1)/10)+5, 0, (numpy.amax(img_mat_2)/10)+5])
        plt.show()
        
def ProcessB1map(Image,TargetFA):
    
    import nibabel as nib
    import numpy
    from scipy import ndimage as ndimage
    from visualization import PlotReconstructedImage
    Image = nib.load(Image)
    Image =Image.get_data()    
    mask=np.zeros(shape=Image.shape)    
    Interp=np.zeros(shape=Image.shape)    
    idx = Image[:,:,:] >= 1
    mask[idx] = 1
    mask=ndimage.binary_dilation(mask,iterations=3)
    PlotReconstructedImage(mask[:,:,3])    
    print(mask.shape)
    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            for k in range(Image.shape[2]):
            
                if mask[i,j,k] == True and Image[i,j,k]==0 :
                    print('Interpolating')
                    if i <Image.shape[0]-1 and j<Image.shape[1]-1 and k < Image.shape[2]-1:
                        # Interp[i,j,k] = numpy.round(Image[i-1,j,k]+Image[i+1,j,k]+Image[i,j-1,k]+Image[i,j+1,k]+Image[i,j,k-1]+Image[i,j,k+1])/6
                        Interp[i,j,k] = numpy.round(Image[i-1,j,k]+Image[i+1,j,k]+Image[i,j-1,k]+Image[i,j+1,k]+Image[i,j,k-1]+Image[i,j,k+1])/6
                    if i <Image.shape[0]-1 and j<Image.shape[1]-1  and k == Image.shape[2]-1:    
                        Interp[i,j,k] = numpy.round(Image[i-1,j,k]+Image[i+1,j,k]+Image[i,j-1,k]+Image[i,j+1,k]+Image[i,j,k-1])/6
                else :
                    Interp[i,j,k] = Image[i,j,k]
    PlotReconstructedImage(Interp[:,:,3])
    PlotReconstructedImage(Image[:,:,3])
    
    RatioMap = np.zeros(shape=Image.shape)    
    RatioMap = (Interp/10)/TargetFA
    
    from nifty_funclib import SaveArrayAsNIfTI
    SaveArrayAsNIfTI(RatioMap,1,1,1,'RatioMap.nii')
    
def ComputeRatioMap(Image,TargetFA):

    import nibabel as nib
    Image = nib.load(Image)
    Image =Image.get_data()    
    RatioMap = np.zeros(shape=Image.shape,dtype=np.float32)    
    RatioMap = (Image/10)/float(TargetFA)
    from nifty_funclib import SaveArrayAsNIfTI
    SaveArrayAsNIfTI(RatioMap,1,1,1,'RatioMap.nii')
    
def ComputeT1map(RatioMap,FA1,Img_FA1,FA2,Img_FA2,mask):    
# def ComputeT1map2(RatioMap,FA1,Img_FA1,FA2,Img_FA2,mask):    

    import nibabel as nib
    import numpy
    from visualization import PlotReconstructedImage
    RatioMap = nib.load(RatioMap)
    RatioMap =RatioMap.get_data()
    Img_FA1 = nib.load(Img_FA1)
    Img_FA1 =Img_FA1.get_data()    
    Img_FA2 = nib.load(Img_FA2)
    Img_FA2 =Img_FA2.get_data()
    mask = nib.load(mask)
    mask =mask.get_data()
    TR=0.1
    FAMap1 = RatioMap*float(FA1)
    FAMap2 = RatioMap*float(FA2)
    print(FAMap1.shape)
    print(FAMap2.shape)
    PlotReconstructedImage((FAMap1[:,:,0,0]))
    PlotReconstructedImage((FAMap2[:,:,0,0]))
    FAMap1=numpy.squeeze(FAMap1)
    FAMap2=numpy.squeeze(FAMap2)
    mask=numpy.squeeze(mask)
    print(FAMap1.shape)
    print(FAMap2.shape)
    # PlotReconstructedImage((mask[:,:]))
    # PlotReconstructedImage((Img_FA1[:,:]))
    # PlotReconstructedImage((Img_FA2[:,:]))
    
    # from nifty_funclib import SaveArrayAsNIfTI
    # SaveArrayAsNIfTI(FAMap1,1,1,1,'MAP1.nii')
    # SaveArrayAsNIfTI(FAMap2,1,1,1,'MAP2.nii')
    
    T1map=np.zeros(shape=Img_FA1.shape, dtype=np.float32)
    for i in range(Img_FA1.shape[0]):
        for j in range(Img_FA1.shape[1]):
            if mask[i,j] != 0:
                S1=float(Img_FA1[i][j])
                S2=float(Img_FA2[i][j])
                C = float((S1*np.sin(FAMap2[i][j]*np.pi/180))/(S2*np.sin(FAMap1[i][j]*np.pi/180)))
                T1map[i][j]=float((-TR)/np.log((C-1)/((np.cos(FAMap1[i][j]*np.pi/180)*C)-np.cos(FAMap2[i][j]*np.pi/180))))
                if T1map[i][j] <=0 : T1map[i][j]=0
                if T1map[i][j] >120 : T1map[i][j]=0

    PlotReconstructedImage((T1map[:,:,0,0]))
    from nifty_funclib import SaveArrayAsNIfTI
    SaveArrayAsNIfTI(T1map,1,1,1,'T1map.nii')
    
def LogSpectralDistance(Image1,Image2):
    import nibabel as nib
    import numpy
    from visualization import PlotImgMag2,PlotReconstructedImage
    Image1 = nib.load(Image1)
    Image1 =Image1.get_data()    
    Image2 = nib.load(Image2)
    Image2 =Image2.get_data()
    Image1 =numpy.squeeze(Image1)
    Image2 =numpy.squeeze(Image2)
    
    Image1_FFT= numpy.zeros(shape=Image1.shape)
    Image1_FFT= numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(Image1)))
    Image2_FFT= numpy.zeros(shape=Image1.shape)
    Image2_FFT= numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(Image2)))
    # PlotImgMag2(numpy.absolute(Image1_FFT))
    # PlotImgMag2(numpy.absolute(Image2_FFT))
    
    
    DLS = numpy.sqrt(float(1/(2*numpy.pi))*sum(sum(10*numpy.log(numpy.absolute(Image1_FFT)**2/(numpy.absolute(Image2_FFT)**2))**2))*(1/(float(Image1.shape[0]*Image1.shape[1]))))
    print(DLS)
    
def ComputeSensitivity(Image,FAmap,T1):

    import nibabel as nib
    import numpy
    from visualization import PlotImgMag2,PlotReconstructedImage
    Image = nib.load(Image)
    Image =Image.get_data()    
    FAmap = nib.load(FAmap)
    FAmap =FAmap.get_data()
    T1 = nib.load(T1)
    T1 =T1.get_data()
    Image =numpy.squeeze(Image)
    FAmap =numpy.squeeze(FAmap)
    T1 =numpy.squeeze(T1)
    
    B1sens = numpy.zeros(shape=Image.shape)
    E1 = numpy.exp(-0.1/T1)
    B1sens = Image/(numpy.sin((30*FAmap)*numpy.pi/180)*((1-E1))/(1-E1*numpy.cos((30*FAmap)*numpy.pi/180)))
    PlotReconstructedImage(B1sens)
    from nifty_funclib import SaveArrayAsNIfTI
    SaveArrayAsNIfTI(B1sens,1,1,1,'B1sens.nii')

def RIO(Img):
    import nibabel as nib
    import numpy
    Img = nib.load(Img)
    Img =Img.get_data()
    from nifty_funclib import SaveArrayAsNIfTI
    SaveArrayAsNIfTI(Img,1,1,1,'Img.nii')    
    
def SegmentationDICE(Seg1,Seg2,label):

    import nibabel as nib
    import numpy
    from visualization import PlotImgMag2,PlotReconstructedImage
    Seg1 = nib.load(Seg1)
    Seg1 =Seg1.get_data()    
    Seg2 = nib.load(Seg2)
    Seg2 =Seg2.get_data()
    # PlotReconstructedImage(Seg1[0,:,:])
    # PlotReconstructedImage(Seg1[:,:])
    # PlotReconstructedImage(Seg2[0,:,:])
    # PlotReconstructedImage(Seg2[:,:])
    if (Seg1.shape != Seg2.shape):
            print('ERROR : Images of different size !')
    else:
    
        intersect_cpt=0
        Seg1LabelCount=0
        Seg2LabelCount=0
        DICE=0.0
        # KB line 2 FISTA line 1        due to different image orientations
        for i in range (Seg1.shape[1]):
        # for i in range (Seg1.shape[0]):
            for j in range(Seg1.shape[2]):
            # for j in range(Seg1.shape[1]):
                # print Seg1[0,i,j]
                # print Seg2[0,i,j]
                if Seg1[0,i,j]==Seg2[0,i,j] and Seg1[0,i,j]==label and Seg2[0,i,j]==label:
                # if Seg1[i,j]==Seg2[i,j] and Seg1[i,j]==label and Seg2[i,j]==label:
                    intersect_cpt+=1
                if Seg1[0,i,j]==label:
                # if Seg1[i,j]==label:
                    Seg1LabelCount+=1
                if Seg2[0,i,j]==label:
                # if Seg2[i,j]==label:
                    Seg2LabelCount+=1
        DICE = float((2.0*int(intersect_cpt))/int((Seg1LabelCount+Seg2LabelCount)))
        print(DICE)
        return DICE
        
# def ImageCorrection(Image, Sensitivity):
def ImageCorrection(Image, Bstatic, Sensitivity,FA,T1):
    
    # GRE Signal : S(r)=(M0*B1-(r)*Rho(r)*1-e^(-TR/T1(r))sin(gamma*B1+(r)*tau)*e^(-TE/T2*(r)))/(1-e^(-TR/T1(r))cos(gamma*B1+(r)*tau))
    # Goal ==> Extract Rho(r)
    # Needed : B1-(r), T1(r), T2*(r), B1+(r) or FA(r) = gamma*B1+(r)*tau
    import nibabel as nib
    import numpy
    from visualization import PlotImgMag2,PlotReconstructedImage
    
    #load acqusisitions
    Image = nib.load(Image)
    Image =Image.get_data()
    Bstatic = nib.load(Bstatic)
    Bstatic =Bstatic.get_data()        
    Sensitivity = nib.load(Sensitivity)
    Sensitivity =Sensitivity.get_data()
    FA = nib.load(FA)
    FA =FA.get_data()    
    T1 = nib.load(T1)
    T1 =T1.get_data()        
    Image=numpy.squeeze(Image)
    Bstatic=numpy.squeeze(Bstatic)
    Sensitivity=numpy.squeeze(Sensitivity)
    FA = numpy.squeeze(FA)
    T1 = numpy.squeeze(T1)
    print(Image.shape)
    
    # Create binary mask of the object and mask image
    mask = numpy.zeros(shape = Image.shape)
    # threshold = 5e-06
    threshold = numpy.mean(Image)
    for i in range(Image.shape[1]):
        for ii in range (Image.shape[0]):
            if Image[i,ii] > threshold:
                mask[i,ii] = 1
            if Image[i,ii] < threshold:
                Image[i,ii]=0
                
    # PlotReconstructedImage(mask)
    # PlotReconstructedImage(Image)
    # PlotReconstructedImage(Sensitivity)        
    # PlotReconstructedImage(FA)    
    # PlotReconstructedImage((T1<35))
    
    # Normalize image intensities between 0 and 1
    Image = Image/numpy.amax(Image)
    Sensitivity= Sensitivity/numpy.amax(Sensitivity)
    Bstatic = Bstatic/numpy.amax(Bstatic)
    FA=30*FA
    TR=0.1
    # Correct for B1- by dividing image with sensitivity profile 
    # Sensi_corected_image = Image/Sensitivity
    # print numpy.amin(Sensi_corected_image),numpy.amax(Sensi_corected_image)
    # PlotReconstructedImage(Sensi_corected_image)
    
    # Then we need to correct for FA and T1 distribution (we load image and use it in equation)
    E1 = numpy.zeros(shape=T1.shape)
    for i in range (T1.shape[0]):
        for ii in range (T1.shape[1]):
            if T1[i,ii] > 0.0 and not numpy.isnan(T1[i,ii]):
                E1[i,ii] = numpy.exp(-TR/T1[i,ii])
            else :
                E1[i,ii] = 1
    RHOM = (Image*(1-E1*numpy.cos(FA)))/(Sensitivity*(1-E1)*numpy.sin(FA))
    PlotReconstructedImage(RHOM)
    from nifty_funclib import SaveArrayAsNIfTI
    SaveArrayAsNIfTI(Image/Sensitivity,1,1,1,'Immasked.nii')
    SaveArrayAsNIfTI(Image/Sensitivity,1,1,1,'Im-S=B1sensi.nii')
    SaveArrayAsNIfTI(RHOM,1,1,1,'RHOM.nii')
    
def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6
    
def ComputeB0(Phase_GRE1,TE1,Phase_GRE2,TE2,mask):

    import nibabel as nib
    import numpy
    from visualization import PlotImgMag2,PlotReconstructedImage
    
    Phase_GRE1 = nib.load(Phase_GRE1)
    Phase_GRE1 =Phase_GRE1.get_data()
    Phase_GRE2 = nib.load(Phase_GRE2)
    Phase_GRE2 =Phase_GRE2.get_data()
    
    if mask==[]:
        mask=numpy.ones(shape=(XFL2.shape))
    else :
        mask = nib.load(mask)
        mask = mask.get_data()
        mask = numpy.squeeze(mask)
    
    # B0=numpy.zeros(shape=GRE1.shape)
    
    PlotReconstructedImage(numpy.unwrap(Phase_GRE1)*mask)
    PlotReconstructedImage(numpy.unwrap(Phase_GRE2)*mask)
    
    B0= (Phase_GRE1-Phase_GRE2)*mask
    gamma = 17235000.0
    DeltaB=B0/(gamma*numpy.absolute(TE2-TE1)*0.001)
    PlotReconstructedImage(B0)
    PlotReconstructedImage(DeltaB)
    from nifty_funclib import SaveArrayAsNIfTI
    SaveArrayAsNIfTI(B0,1,1,1,'B0_test.nii')
    return B0
    
def SubSampleImg(Image,outReso,weight,save):
    
    import numpy
    import nibabel as nib
    from visualization import PlotReconstructedImage
    
    Image = nib.load(Image)
    Image =Image.get_data()
    
    if save==[]:
        Save=False
    else:
        Save=True

    print("Input  Image size : ", Image.shape[0]," x ", Image.shape[1])
    print("Output Image size : ", int(outReso), " x ", int(outReso))
    
    SubSampledImage=numpy.zeros(shape=(int(outReso),int(outReso)))
    
    blocksize=Image.shape[0]/int(outReso)
    
    if weight == [] :
        print("No Weighting applied just taking mean of blocks")
        for i in range(int(outReso)):
            for j in range(int(outReso)):
                SubSampledImage[i,j] = numpy.mean(Image[i*blocksize:(i+1)*blocksize,j*blocksize:(j+1)*blocksize])
    
    PlotReconstructedImage(numpy.squeeze(Image))
    PlotReconstructedImage(SubSampledImage)
    
    if Save:
        from nifty_funclib import SaveArrayAsNIfTI
        SaveArrayAsNIfTI(SubSampledImage,1,1,1,'SubsampledImage.nii')
    
    return SubSampledImage
    
def ComputeB1emit(XFL1,XFL2,mask,targetangle,save):
    
    import numpy
    import nibabel as nib
    from visualization import PlotReconstructedImage
    
    XFL1 = nib.load(XFL1)
    XFL1 = XFL1.get_data()
    XFL1 = numpy.squeeze(XFL1)
    XFL2 = nib.load(XFL2)
    XFL2 = XFL2.get_data()
    XFL2 = numpy.squeeze(XFL2)
    
    print(XFL1.shape)
    print(XFL2.shape)
    
    if save==[]:
        Save=False
    else:
        Save=True
    
    if mask==[]:
        mask=numpy.ones(shape=(XFL2.shape))
    else :
        mask = nib.load(mask)
        mask = mask.get_data()
        mask = numpy.squeeze(mask)
    
    degImg = numpy.zeros(shape=(XFL2.shape))
    for i in range(XFL2.shape[0]):
        for j in range(XFL2.shape[1]):
            if (float(XFL1[i,j])==0):
                degImg[i,j]=0
            else:
                degImg[i,j]=numpy.arccos(float(XFL2[i,j])/float(XFL1[i,j]))*(180/numpy.pi)*mask[i,j]
    ratiomap = degImg/targetangle
    
    PlotReconstructedImage(degImg)
    PlotReconstructedImage(ratiomap)
    if save:
        from nifty_funclib import SaveArrayAsNIfTI
        SaveArrayAsNIfTI(ratiomap,1,1,1,'ratiomap_test.nii')
    
    return ratiomap
    
def ComputeB1sensCorrectedImage(InputImg,InterpolatedFiedl,mask,save):

    import numpy
    import nibabel as nib
    from visualization import PlotReconstructedImage
    
    if save==[]:
        Save=False
    else:
        Save=True
 
    InputImg = nib.load(InputImg)
    InputImg = InputImg.get_data()
    InterpolatedFiedl = nib.load(InterpolatedFiedl)
    InterpolatedFiedl = InterpolatedFiedl.get_data()
    
    if mask==[]:
        mask=numpy.ones(shape=(InputImg.shape))
    else :
        mask = nib.load(mask)
        mask = mask.get_data()
        mask = numpy.squeeze(mask)
        
    PlotReconstructedImage(mask)    
    PlotReconstructedImage(InputImg)    
    PlotReconstructedImage(InterpolatedFiedl)    
        
    B1sens = numpy.zeros(shape=(InputImg.shape))
    for i in range(InputImg.shape[0]):
        for j in range(InputImg.shape[1]):
            if (float(mask[i,j])==0.0):
                B1sens[i,j]=0.0
            else:
                B1sens[i,j]=float(InputImg[i,j])/float(InterpolatedFiedl[i,j])
                
    PlotReconstructedImage(B1sens)

    if save:
        from nifty_funclib import SaveArrayAsNIfTI
        SaveArrayAsNIfTI(B1sens,1,1,1,'B1sensCorrected.nii')
    
    return B1sens    
    
    
# def ComputeXDensity(RatioMap,FA1,Img_FA1,FA2,Img_FA2,FA3,Img_FA3,FA4,Img_FA4,mask):
def ComputeXDensity(RatioMap,FA1,Img_FA1,FA2,Img_FA2,mask):

    import nibabel as nib
    import numpy
    from scipy import stats
    from visualization import PlotReconstructedImage
    RatioMap = nib.load(RatioMap)
    RatioMap =RatioMap.get_data()
    Img_FA1 = nib.load(Img_FA1)
    Img_FA1 =Img_FA1.get_data()    
    Img_FA2 = nib.load(Img_FA2)
    Img_FA2 =Img_FA2.get_data()
    # Img_FA3 = nib.load(Img_FA3)
    # Img_FA3 =Img_FA3.get_data()
    # Img_FA4 = nib.load(Img_FA4)
    # Img_FA4 =Img_FA4.get_data()
    mask = nib.load(mask)
    mask =mask.get_data()
    TR=0.1
    FAMap1 = RatioMap*float(FA1)
    FAMap2 = RatioMap*float(FA2)
    # FAMap3 = RatioMap*float(FA3)
    # FAMap4 = RatioMap*float(FA4)
    
    print(FAMap1.shape)
    print(FAMap2.shape)
    # PlotReconstructedImage((FAMap1[:,:,0,0]))
    # PlotReconstructedImage((FAMap2[:,:,0,0]))
    FAMap1=numpy.squeeze(FAMap1)
    FAMap2=numpy.squeeze(FAMap2)
    # FAMap3=numpy.squeeze(FAMap3)
    # FAMap4=numpy.squeeze(FAMap4)
    
    mask=numpy.squeeze(mask)
    # PlotReconstructedImage((mask[:,:]))
    # PlotReconstructedImage((Img_FA1[:,:]))
    # PlotReconstructedImage((Img_FA2[:,:]))
    
    T1 = numpy.zeros(shape=Img_FA1.shape)
    M0 = numpy.zeros(shape=Img_FA1.shape)
    
    for i in range(len(FAMap1[0])):
        for j in range(len(FAMap1[0])):
            if mask[i,j]==1:
                y = numpy.zeros(2)
                x = numpy.zeros(2)
                # y[3,0]=Img_FA1[i,j]/numpy.sin(FAMap1[i,j]*numpy.pi/180.0)
                # y[2,0]=Img_FA2[i,j]/numpy.sin(FAMap2[i,j]*numpy.pi/180.0)
                # y[1,0]=Img_FA3[i,j]/numpy.sin(FAMap3[i,j]*numpy.pi/180.0)
                # y[0,0]=Img_FA4[i,j]/numpy.sin(FAMap4[i,j]*numpy.pi/180.0)
                # y[3,1]=Img_FA1[i,j]/numpy.tan(FAMap1[i,j]*numpy.pi/180.0)
                # y[2,1]=Img_FA2[i,j]/numpy.tan(FAMap2[i,j]*numpy.pi/180.0)
                # y[1,1]=Img_FA3[i,j]/numpy.tan(FAMap3[i,j]*numpy.pi/180.0)
                # y[0,1]=Img_FA4[i,j]/numpy.tan(FAMap4[i,j]*numpy.pi/180.0)
                
                # y[1,0]=Img_FA1[i,j]/numpy.sin(FAMap1[i,j]*numpy.pi/180.0)
                # y[0,0]=Img_FA2[i,j]/numpy.sin(FAMap2[i,j]*numpy.pi/180.0)
                # y[1,1]=Img_FA1[i,j]/numpy.tan(FAMap1[i,j]*numpy.pi/180.0)
                # y[0,1]=Img_FA2[i,j]/numpy.tan(FAMap2[i,j]*numpy.pi/180.0)
                y[1]=Img_FA1[i,j]/numpy.sin(FAMap1[i,j]*numpy.pi/180.0)
                y[0]=Img_FA2[i,j]/numpy.sin(FAMap2[i,j]*numpy.pi/180.0)
                x[1]=Img_FA1[i,j]/numpy.tan(FAMap1[i,j]*numpy.pi/180.0)
                x[0]=Img_FA2[i,j]/numpy.tan(FAMap2[i,j]*numpy.pi/180.0)
                # print y
                # slope, intercept, r_value, p_value, std_err = stats.linregress(y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                print(r_value**2, p_value, std_err)
                # slope = (y[1]-y[0])/(x[1]-x[0])
                # intercept = y[0]-slope*x[0]
                T1[i,j]=-TR/numpy.log(slope)
                M0[i,j]=intercept/((1-slope)*numpy.exp(-0.1e-3/12e-3))

            else :
                T1[i,j]=0
                M0[i,j]=0
    # PlotReconstructedImage((T1[:,:]))
    # PlotReconstructedImage((M0[:,:]))    
    from nifty_funclib import SaveArrayAsNIfTI
    SaveArrayAsNIfTI(T1,1,1,1,'testT1_papier.nii')    
    SaveArrayAsNIfTI(M0,1,1,1,'testM0_papier.nii')    
    
def NormalizeImageIntensity(Imagepath):

    import nibabel as nib
    import os, sys, numpy
    
    Image = nib.load(Imagepath)
    Image =Image.get_data()
    print(Image.shape)
    Image = numpy.squeeze(Image)
    min, max = MinMaxImage(Image)
    NormalizedImage=np.zeros(Image.shape)
    for i in range(Image.shape[0]):
            for j in range (Image.shape[1]):
                if len(Image.shape)==3: 
                    for k in range(Image.shape[2]):
                        NormalizedImage[i,j,k] = Image[i,j,k]/max
                else:
                    NormalizedImage[i,j] = Image[i,j]/max
    from nifty_funclib import SaveArrayAsNIfTI
    Hpath, Fname = os.path.split(Imagepath)
    Fname = Fname.split('.')
    OutputPath = os.path.join( Hpath + '\\' + Fname[0] + '_Normalized.nii')
    SaveArrayAsNIfTI(NormalizedImage,1,1,1,OutputPath)    

def RemoveNaN(Imagepath):

    import nibabel as nib
    import numpy, os
    
    Image = nib.load(Imagepath)
    Image =Image.get_data()
    Image = numpy.squeeze(Image)
    NoNaNImage=np.zeros(Image.shape)
    for i in range(Image.shape[0]):
            for j in range (Image.shape[1]):
                if len(Image.shape)==3: 
                    for k in range(Image.shape[2]):
                        if numpy.isnan(Image[i,j,k]):
                            NoNaNImage[i,j,k] = 0
                        else:
                            NoNaNImage[i,j,k]=Image[i,j,k]
                else:
                    if numpy.isnan(Image[i,j]):
                            NoNaNImage[i,j] = 0
                    else:
                        NoNaNImage[i,j]=Image[i,j]

    from nifty_funclib import SaveArrayAsNIfTI
    Hpath, Fname = os.path.split(Imagepath)
    Fname = Fname.split('.')
    OutputPath = os.path.join( Hpath + '\\' + Fname[0] + '_NaNRemoved.nii')
    SaveArrayAsNIfTI(NoNaNImage,1,1,1,OutputPath)
    
def ComputeXDensity3D(RatioMap,FA1,Img_FA1,FA2,Img_FA2,mask):

    import nibabel as nib
    import numpy, os, sys
    from scipy import stats
    from visualization import PlotReconstructedImage
    path=RatioMap
    RatioMap = nib.load(RatioMap)
    RatioMap =RatioMap.get_data()
    Img_FA1 = nib.load(Img_FA1)
    Img_FA1 =Img_FA1.get_data()    
    Img_FA2 = nib.load(Img_FA2)
    Img_FA2 =Img_FA2.get_data()
    # Img_FA3 = nib.load(Img_FA3)
    # Img_FA3 =Img_FA3.get_data()
    # Img_FA4 = nib.load(Img_FA4)
    # Img_FA4 =Img_FA4.get_data()
    mask = nib.load(mask)
    mask =mask.get_data()
    TR=0.02                #Sodium        
    # TR=0.0084                # Proton (values from Sabati's paper)
    TE=0.0001                #Sodium
    # TE=0.00376                #Proton     (values from Sabati's paper)
    T2star=0.012
    # T2star=0.3
    print(FA1, FA2)
    FAMap1 = RatioMap*float(FA1)
    FAMap2 = RatioMap*float(FA2)
    # FAMap3 = RatioMap*float(FA3)
    # FAMap4 = RatioMap*float(FA4)

    FAMap1=numpy.squeeze(FAMap1)
    FAMap2=numpy.squeeze(FAMap2)
    mask=numpy.squeeze(mask)
    print(FAMap1.shape)
    print(FAMap2.shape)
    print(mask.shape)
    # PlotReconstructedImage((mask[:,:]))
    # PlotReconstructedImage((Img_FA1[:,:]))
    # PlotReconstructedImage((Img_FA2[:,:]))
    
    T1 = numpy.zeros(shape=Img_FA1.shape)
    M0 = numpy.zeros(shape=Img_FA1.shape)
    
    # for i in range(len(FAMap1[2])):
        # for j in range(len(FAMap1[1])):
            # for k in range(len(FAMap1[0])):
    for i in range(Img_FA1.shape[0]):
        for j in range(Img_FA1.shape[1]):
            for k in range(Img_FA1.shape[2]):
                if mask[i,j,k]>0:
                    y = numpy.zeros(2)
                    x = numpy.zeros(2)
                
                    y[0]=Img_FA1[i,j,k]/numpy.sin(FAMap1[i,j,k]*numpy.pi/180.0)
                    y[1]=Img_FA2[i,j,k]/numpy.sin(FAMap2[i,j,k]*numpy.pi/180.0)
                    x[0]=Img_FA1[i,j,k]/numpy.tan(FAMap1[i,j,k]*numpy.pi/180.0)
                    x[1]=Img_FA2[i,j,k]/numpy.tan(FAMap2[i,j,k]*numpy.pi/180.0)
                    # print y
                    # slope, intercept, r_value, p_value, std_err = stats.linregress(y)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                    # print r_value**2, p_value, std_err
                    # slope = (y[1]-y[0])/(x[1]-x[0])
                    # intercept = y[0]-slope*x[0]
                    if not numpy.isnan(-TR/numpy.log(slope)):
                        T1[i,j,k]=-TR/numpy.log(slope)
                    else:
                        T1[i,j,k]=0
                    M0[i,j,k]=intercept/((1-slope)*numpy.exp(-TE/T2star))

            else :
                T1[i,j,k]=0
                M0[i,j,k]=0
    # PlotReconstructedImage((T1[:,:]))
    # PlotReconstructedImage((M0[:,:]))    
    from nifty_funclib import SaveArrayAsNIfTI
    Hpath, Fname = os.path.split(str(path))
    Fname = Fname.split('.')
    OutputPathT1 = os.path.join( Hpath + '\\' + "T1-3D.nii")
    OutputPathM0 = os.path.join( Hpath + '\\' + "M0-3D.nii")
    SaveArrayAsNIfTI(T1,1,1,1,OutputPathT1)    
    SaveArrayAsNIfTI(M0,1,1,1,OutputPathM0)    
        
        
def ComputeXDensity3D_3pts(RatioMap,FA1,Img_FA1,FA2,Img_FA2,FA3,Img_FA3,mask):

    import nibabel as nib
    import numpy, os, sys
    from scipy import stats
    from visualization import PlotReconstructedImage
    RatioMap = nib.load(RatioMap)
    RatioMap =RatioMap.get_data()
    Img_FA1 = nib.load(Img_FA1)
    Img_FA1 =Img_FA1.get_data()    
    Img_FA2 = nib.load(Img_FA2)
    Img_FA2 =Img_FA2.get_data()
    Img_FA3 = nib.load(Img_FA3)
    Img_FA3 =Img_FA3.get_data()
    # Img_FA4 = nib.load(Img_FA4)
    # Img_FA4 =Img_FA4.get_data()
    mask = nib.load(mask)
    mask =mask.get_data()
    TR=0.02
    TE=0.0001
    T2star=0.012
    FAMap1 = RatioMap*float(FA1)
    FAMap2 = RatioMap*float(FA2)
    FAMap3 = RatioMap*float(FA3)
    # FAMap4 = RatioMap*float(FA4)
    
    print(FAMap1.shape)
    print(FAMap2.shape)
    # PlotReconstructedImage((FAMap1[:,:,0,0]))
    # PlotReconstructedImage((FAMap2[:,:,0,0]))
    FAMap1=numpy.squeeze(FAMap1)
    FAMap2=numpy.squeeze(FAMap2)
    # FAMap3=numpy.squeeze(FAMap3)
    # FAMap4=numpy.squeeze(FAMap4)
    
    mask=numpy.squeeze(mask)
    # PlotReconstructedImage((mask[:,:]))
    # PlotReconstructedImage((Img_FA1[:,:]))
    # PlotReconstructedImage((Img_FA2[:,:]))
    
    T1 = numpy.zeros(shape=Img_FA1.shape)
    M0 = numpy.zeros(shape=Img_FA1.shape)
    
    for i in range(Img_FA1.shape[0]):
        for j in range(Img_FA1.shape[1]):
            for k in range(Img_FA1.shape[2]):
                if mask[i,j,k]==1:
                    y = numpy.zeros(3)
                    x = numpy.zeros(3)
                    
                    y[2]=Img_FA3[i,j,k]/numpy.sin(FAMap3[i,j,k]*numpy.pi/180.0)
                    y[1]=Img_FA2[i,j,k]/numpy.sin(FAMap2[i,j,k]*numpy.pi/180.0)
                    y[0]=Img_FA1[i,j,k]/numpy.sin(FAMap1[i,j,k]*numpy.pi/180.0)
                    x[2]=Img_FA3[i,j,k]/numpy.tan(FAMap3[i,j,k]*numpy.pi/180.0)
                    x[1]=Img_FA2[i,j,k]/numpy.tan(FAMap2[i,j,k]*numpy.pi/180.0)
                    x[0]=Img_FA1[i,j,k]/numpy.tan(FAMap1[i,j,k]*numpy.pi/180.0)
                    # print y
                    # slope, intercept, r_value, p_value, std_err = stats.linregress(y)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                    print(r_value**2, p_value, std_err)
                    # slope = (y[1]-y[0])/(x[1]-x[0])
                    # intercept = y[0]-slope*x[0]
                    T1[i,j,k]=-TR/numpy.log(slope)
                    M0[i,j,k]=intercept/((1-slope)*numpy.exp(-TE/T2star))

            else :
                T1[i,j,k]=0
                M0[i,j,k]=0
    # PlotReconstructedImage((T1[:,:]))
    # PlotReconstructedImage((M0[:,:]))    
    from nifty_funclib import SaveArrayAsNIfTI
    Hpath, Fname = os.path.split(str(path))
    Fname = Fname.split('.')
    OutputPathT1 = os.path.join( Hpath + '\\' + "T1-3D.nii")
    OutputPathM0 = os.path.join( Hpath + '\\' + "M0-3D.nii")
    SaveArrayAsNIfTI(T1,1,1,1,OutputPathT1)    
    SaveArrayAsNIfTI(M0,1,1,1,OutputPathM0)    


def MeanStdCalc(Image):

    import nibabel as nib
    from scipy import stats
    Image = nib.load(Image)
    Image =Image.get_data()
    sum=0
    pts=0
    values=[]
    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            for k in range(Image.shape[2]):
                if(Image[i,j,k]>0 and not np.isnan( Image[i,j,k]) and Image[i,j,k]<500):
                    sum += float(Image[i,j,k])
                    pts += 1
                    values.append(float(Image[i,j,k]))
    mean = sum/pts
    print("mean = ", mean, end=' ') 
    print("stdev = ", np.std(values))
    print("skewness = ", stats.skew(values, bias=True))
    print("kurtosis = ", stats.kurtosis(values, fisher=True, bias=True))

    import matplotlib.pyplot as plt
    plt.hist(values, bins=100) 
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    
def MeanStdCalc2(Image,plot,maxbound):

    import nibabel as nib
    from scipy import stats
    Image = nib.load(Image)
    Image =Image.get_data()
    sum=0
    pts=0
    values=[]
    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            for k in range(Image.shape[2]):
                if( np.isnan( Image[i,j,k])==False and np.isinf(Image[i,j,k])==False and Image[i,j,k]>0 and Image[i,j,k]<maxbound):
                    sum += float(Image[i,j,k])
                    pts += 1
                    values.append(float(Image[i,j,k]))
    mean = sum/pts
    print("mean = ", mean, end=' ') 
    print("stdev = ", np.std(values))
    print("skewness = ", stats.skew(values, bias=True))
    print("kurtosis = ", stats.kurtosis(values, fisher=True, bias=True))
    print("min value = ", np.amin(values))
    print("max value = ", np.amax(values))
    if plot==1:
        import matplotlib.pyplot as plt
        plt.hist(values, bins=100) 
        plt.title("Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()    
    return values
    
def MeanStdCalc3(Image,plot,maxbound):

    import nibabel as nib
    from scipy import stats
    # Image = nib.load(Image)
    # Image =Image.get_data()
    sum=0
    pts=0
    values=[]
    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            for k in range(Image.shape[2]):
                if( np.isnan( Image[i,j,k])==False and np.isinf(Image[i,j,k])==False and Image[i,j,k]>0 and Image[i,j,k]<maxbound):
                    sum += float(Image[i,j,k])
                    pts += 1
                    values.append(float(Image[i,j,k]))
    mean = sum/pts
    print("mean = ", mean, end=' ') 
    print("stdev = ", np.std(values))
    print("skewness = ", stats.skew(values, bias=True))
    print("kurtosis = ", stats.kurtosis(values, fisher=True, bias=True))
    print("min value = ", np.amin(values))
    print("max value = ", np.amax(values))
    if plot==1:
        import matplotlib.pyplot as plt
        plt.hist(values, bins=100) 
        plt.title("Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()    
    return values,mean,np.std(values)    
    
def HistogramCompare(Image1,Image2):
    import nibabel as nib
    from scipy import stats
    Image1 = nib.load(Image1)
    Image1 =Image1.get_data()
    Image2 = nib.load(Image2)
    Image2 =Image2.get_data()

    values1=[]
    values2=[]
    for i in range(Image1.shape[0]):
        for j in range(Image1.shape[1]):
            for k in range(Image1.shape[2]):
                if(Image1[i,j,k]>0 and not np.isnan( Image1[i,j,k]) and Image1[i,j,k]<500):
                    values1.append(float(Image1[i,j,k]))
                    values2.append(float(Image2[i,j,k]))
    
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 100, 100)

    plt.hist(values1, bins, alpha=0.5, label='Im1')
    plt.hist(values2, bins, alpha=0.5, label='Im2')
    plt.legend(loc='upper right')
    plt.show()

def HistogramCompare2(Image1,Image2):    

    import nibabel as nib
    import numpy as np
    import matplotlib.mlab as mlab
    from scipy import stats
    Image1 = nib.load(Image1)
    Image1 =Image1.get_data()
    Image2 = nib.load(Image2)
    Image2 =Image2.get_data()

    values1=[]
    values2=[]
    for i in range(Image1.shape[0]):
        for j in range(Image1.shape[1]):
            for k in range(Image1.shape[2]):
                if(Image1[i,j,k]>0 and not np.isnan( Image1[i,j,k]) and Image1[i,j,k]<500):
                    values1.append(float(Image1[i,j,k]))
                    values2.append(float(Image2[i,j,k]))
    m1=np.mean(values1)
    m2=np.mean(values2)
    std1=np.std(values1)
    std2=np.std(values2)
    print("mean I1 =",m1," +/- ",std1)
    print("mean I2 =",m2," +/- ",std2)
    import matplotlib.pyplot as plt
    bins = np.linspace(20, 120, 100)
    label=['TSC', 'VaSCo']
    plt.figure(1)
    
    plt.subplot(311)
    plt.title( "Number of voxels =  %s \n TSC = %s +/- %s \n VaSCo = %s +/- %s " % (len(values1),m1,std1,m2,std2) )
    plt.hist(values1, bins, normed=1, histtype='bar', color=['crimson'], alpha=0.5)
    plt.hist(values2, bins, normed=1, histtype='bar', color=['burlywood'], alpha=0.5)
    plt.legend(label)

    plt.subplot(312)
    plt.hist(values1, bins=bins, normed=1, histtype='step', cumulative=1)
    plt.hist(values2, bins=bins, normed=1, histtype='step', cumulative=1)
    plt.legend(label,loc='upper right')
    plt.subplot(313)
    y1 = mlab.normpdf(bins, m1, std1)
    y2 = mlab.normpdf(bins, m2, std2)
    plt.plot(bins, y1, 'r--')
    plt.plot(bins, y2, 'b--')
    plt.legend(label,loc='upper right')
    
    plt.show()
    
    
def AffineInterpNaConcentration(NoiseMean,TubeConc1,TubeVal1,TubeConc2,TubeVal2):

    from scipy import stats
    y = np.zeros(3)
    x = np.zeros(3)

    y[2]=TubeVal2;
    y[1]=TubeVal1;
    y[0]=NoiseMean;
    x[2]=TubeConc2
    x[1]=TubeConc1
    x[0]=0
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y);
    print("Regression Parameters : slope = ",slope, "Intercept = ",intercept, "with R = ",r_value)
    return slope, intercept
    
def AffineInterpNaConcentration2pts(NoiseMean,CSFmean):

    from scipy import stats
    y = np.zeros(2)
    x = np.zeros(2)

    y[1]=CSFmean;
    y[0]=NoiseMean;

    x[1]=145
    x[0]=0
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y);
    print("2 pts Regression Parameters : slope = ",slope, "Intercept = ",intercept, "with R = ",r_value)
    return slope, intercept
    
def AffineInterpNaConcentration2ptsAndCorrection(NoiseMean,CSFmean,Im):

    import numpy,os
    import nibabel as nib
    Im = nib.load(Im)
    Im =Im.get_data()
    from scipy import stats
    y = np.zeros(2)
    x = np.zeros(2)


    y[1]=CSFmean;
    y[0]=NoiseMean;
    
    x[1]=145
    x[0]=0
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y);
    print("2 pts Regression Parameters : slope = ",slope, "Intercept = ",intercept, "with R = ",r_value)
    
    ScaledIm = numpy.zeros(shape=Im.shape)
    for i in range(Im.shape[0]):
        for j in range(Im.shape[1]):
            for k in range(Im.shape[2]):
                if (Im[i,j,k]>0):
                    ScaledIm[i,j,k]=(Im[i,j,k]-intercept)/slope
    
    from nifty_funclib import SaveArrayAsNIfTI
    # Hpath, Fname = os.path.split(str(Im))
    # print Hpath, Fname
    # Fname = Fname.split('.')
    # OutputPathScaledIm = os.path.join( Hpath + '\\' + "2ptsCalib.nii")
    SaveArrayAsNIfTI(ScaledIm,4,4,4,"C:/Users/AC243636/Pictures/2ptsCalib.nii")        
    return slope, intercept, ScaledIm
    
def CorrectTSCImage(TSC,B1map,T1,TR):

    import nibabel as nib
    import numpy, os, sys
    
    B1map = nib.load(B1map)
    B1map =B1map.get_data()
    TSC = nib.load(TSC)
    TSC =TSC.get_data()
    # TR=0.120;
    CorrectedTSC= numpy.zeros(shape=TSC.shape)
    E1=numpy.exp(-TR/T1)
    Alpha=3.1415926535/2;
    
    for i in range(TSC.shape[0]):
        for j in range(TSC.shape[1]):
            for k in range(TSC.shape[2]):
            
                CorrectedTSC[i,j,k]=TSC[i,j,k]*((1-(numpy.cos(B1map[i,j,k]*Alpha)*E1))/(1-E1))
                CorrectedTSC[i,j,k]=CorrectedTSC[i,j,k]/B1map[i,j,k]        # Correction for B1-
    
    from nifty_funclib import SaveArrayAsNIfTI
    # Hpath, Fname = os.path.split(str(TSC))
    # print Hpath
    # Fname = Fname.split('.')
    # OutputPath = os.path.join( Hpath + '\\' + "CorrectedTSC.nii")
    # print OutputPath
    SaveArrayAsNIfTI(CorrectedTSC,4,4,4,"C:/Users/AC243636/Pictures/CorrectedTSC_withB1sensibility.nii")    
    
def ComputeRhoSingleIm(Img,FA,RatioMap,T1map):

    import nibabel as nib
    import numpy, os, sys
    from scipy import stats
    from visualization import PlotReconstructedImage
    RatioMap = nib.load(RatioMap)
    RatioMap =RatioMap.get_data()
    Img = nib.load(Img)
    Img =Img.get_data()    
    FA=float(FA)
    T1map = nib.load(T1map)
    T1map =T1map.get_data()
    TR=0.02
    Rho = numpy.zeros(shape=Img.shape)
    
    for i in range(Img.shape[0]):
        for j in range(Img.shape[1]):
            for k in range(Img.shape[2]):
                if T1map[i,j,k]==255:
                    T1=0.05
                if T1map[i,j,k]==218:
                    T1=0.032
                if T1map[i,j,k]==132:
                    T1=0.022
                else:
                    T1=0.05
                ELong=numpy.exp(-TR/T1)
                SIN=numpy.sin(FA*RatioMap[i,j,k]*numpy.pi/180.0)
                COS=numpy.cos(FA*RatioMap[i,j,k]*numpy.pi/180.0)
                Rho[i,j,k]=(Img[i,j,k]*(1-COS*ELong))/(RatioMap[i,j,k]*SIN*(1-ELong))
    
                if numpy.isinf(Rho[i,j,k]):
                    Rho[i,j,k]=0
    
    from nifty_funclib import SaveArrayAsNIfTI
    # Hpath, Fname = os.path.split(str(Img))
    # print Hpath
    # Fname = Fname.split('.')
    # OutputPathRho = os.path.join( Hpath + '\\' + "Rho-3D.nii")
    # print OutputPathRho
    SaveArrayAsNIfTI(Rho,1,1,1,"C:/Users/AC243636/Pictures/TTTTTEEEEESSSTTT.nii")    
            
                
def CompareValuesPlot(Image1,Image2):
    
    import nibabel as nib
    import numpy as np
    import matplotlib.mlab as mlab
    from matplotlib import gridspec
    from scipy.stats import gaussian_kde
    
    from scipy import stats
    Image1 = nib.load(Image1)
    Image1 =Image1.get_data()
    Image2 = nib.load(Image2)
    Image2 =Image2.get_data()

    values1=[]
    values2=[]
    for i in range(Image1.shape[0]):
        for j in range(Image1.shape[1]):
            for k in range(Image1.shape[2]):
                if(Image1[i,j,k]>0 and not np.isnan( Image1[i,j,k]) and Image1[i,j,k]<100 and Image2[i,j,k]<100):
                    values1.append(float(Image1[i,j,k]))
                    values2.append(float(Image2[i,j,k]))
    
    m1=np.mean(values1)
    m2=np.mean(values2)
    std1=np.std(values1)
    std2=np.std(values2)
    print("mean I1 =",m1," +/- ",std1)
    print("mean I2 =",m2," +/- ",std2)
    import matplotlib.pyplot as plt    
    # plt.plot(values1, values2, 'o', alpha=0.3)
    # plt.axis([np.amin([np.amin(values1),np.amin(values2)]), np.amax([np.amax(values1),np.amax(values2)]), np.amin([np.amin(values1),np.amin(values2)]), np.amax([np.amax(values1),np.amax(values2)])])
    slope, intercept, r_value, p_value, std_err = stats.linregress(values2,values1);
    x=np.linspace(np.amin([np.amin(values1),np.amin(values2)]), np.amax([np.amax(values1),np.amax(values2)]), 100)
    y=slope*x+intercept
    
    y1 = mlab.normpdf(x, m1, std1)
    y2 = mlab.normpdf(x, m2, std2)
    fig = plt.figure(figsize=(12, 12)) 
    print(slope, intercept, p_value, r_value, std_err)
    # plt.plot(x,y,'k',linewidth=2)
    # plt.plot(x,x,'g',linewidth=2)
    # plt.title("Concentration distribution comparison")
    # plt.xlabel("TSC")
    # plt.ylabel("VaSCo")
    # plt.show()
    
    # data=np.column_stack((values1,values2))
    xData = np.reshape(values1, (len(values1), 1))
    yData = np.reshape(values2, (len(values2), 1))
    data = np.hstack((xData, yData))
    print(data.shape)

    mu = data.mean(axis=0)
    data = data - mu
    # data = (data - mu)/data.std(axis=0)  # Uncommenting this reproduces mlab.PCA results
    eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
    print(eigenvectors)
    projected_data = np.dot(data, eigenvectors)
    sigma = projected_data.std(axis=0).mean()
    # sigma2 = projected_data.std(axis=1).mean()
    print((eigenvectors, eigenvalues, sigma))

    # fig = plt.figure(figsize=(8, 6)) 
    gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1,4],
                       height_ratios=[4,1]
                       )

    
    ax2 = plt.subplot(gs[1])
    ax1 = plt.subplot(gs[0],sharey=ax2)
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3],sharex=ax2)
    ax2.plot(values1, values2, 'o', alpha=0.3)
    ax2.axis([np.amin([np.amin(values1),np.amin(values2)]), np.amax([np.amax(values1),np.amax(values2)]), np.amin([np.amin(values1),np.amin(values2)]), np.amax([np.amax(values1),np.amax(values2)])])
    # ax2.plot(x,y,'k',linewidth=2)
    i=0
    for axis in eigenvectors:
        start, end = mu, mu + sigma *(eigenvalues[i]/eigenvalues[0]) * axis
        ax2.annotate('', xy=end, xycoords='data',
        xytext=start, textcoords='data',
        arrowprops=dict(facecolor='red', width=2.0))
        i=i+1
        # ax2.arrow(mu, mu, mu + sigma1 * axis[0], m1 + sigma1 * axis[1], head_width=0.5, head_length=0.3, fc='k', ec='k')
    ax2.plot(x,x,'g',linewidth=2)
    # plt.xlabel("TSC")
    # plt.ylabel("VaSCo")
    
    ax1.plot(y2,x)
    ax3.hist2d(values1,values2,(40, 40), cmap=plt.cm.jet)
    ax4.plot(x,y1)
    # plt.title ("WM [Na] Concentration Comparison TSC/VaSCo")

    fig.suptitle ("TSC uncorected/ VaSCo [Na] Concentration comparison in WM ROI \n Number of voxels =  %s \n TSC = %s +/- %s \n VaSCo = %s +/- %s " % (len(values1),m1,std1,m2,std2))
    plt.show()
    
def BoxPlotDistrib(Image1, Image2, Image3, Image4):
    
    import nibabel as nib
    from scipy import stats

    maxbound=0.1
    
    from processingFunctions import MeanStdCalc2
    val1=MeanStdCalc2(Image1,0,maxbound)
    val2=MeanStdCalc2(Image2,0,maxbound)
    val3=MeanStdCalc2(Image3,0,maxbound)
    val4=MeanStdCalc2(Image4,0,maxbound)
    
    import matplotlib.pyplot as plt
    data = [val1,val2,val3,val4]
    # labels = list('B0639','B0643','B0645','B0664')
    # multiple box plots on one figure
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)
    plt.boxplot(data, sym='')
    # ax.set_xticklabels(['B0639','B0643','B0645','B0664'])
    ax.set_xticklabels(['A','B','C','D'])
    bp = ax.boxplot(data, patch_artist=True, notch=True)
    for box in bp['boxes']:
    # change outline color
        
        box.set( color='#222222', linewidth=3)
        # change fill color
        box.set( facecolor = '#FFFFFF' )
    for cap in bp['caps']:
        cap.set(color='#222222', linewidth=3)
    for whisker in bp['whiskers']:
        whisker.set(color='#222222', linewidth=3)
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#222222', linewidth=3)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#222222', alpha=0)
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.title('T1 in homogeneous WM')
    plt.ylabel('T1 in s')
    plt.ylim(0, 0.07)
    plt.show()
    
def BoxPlotCompareDistrib(Image1, Image2, Image3, Image4,
                          Image5, Image6, Image7, Image8):
    
    import nibabel as nib
    from scipy import stats
    from matplotlib import gridspec
    maxbound=120
        
    from processingFunctions import MeanStdCalc2
    val1=MeanStdCalc2(Image1,0,maxbound)
    val2=MeanStdCalc2(Image2,0,maxbound)
    val3=MeanStdCalc2(Image3,0,maxbound)
    val4=MeanStdCalc2(Image4,0,maxbound)
    val5=MeanStdCalc2(Image5,0,maxbound)
    val6=MeanStdCalc2(Image6,0,maxbound)
    val7=MeanStdCalc2(Image7,0,maxbound)
    val8=MeanStdCalc2(Image8,0,maxbound)
    
    import matplotlib.pyplot as plt
    data = [val1,val3,val5,val7,val2,val4,val6,val8]
    # data = [val1,val2,val3,val4,val5,val6,val7,val8]
    # labels = list('B0639','B0643','B0645','B0664')
    # multiple box plots on one figure
    fig = plt.figure(1)
    
    means = [np.mean(data1) for data1 in data]
    length = [np.size(data1) for data1 in data]
    # SDOM = [means[l]/np.sqrt(length[l]) for l in len(length)]
    print(means)
    print(length)
    positions = np.arange(8) + 1
    gs = gridspec.GridSpec(1, 2,
                       width_ratios=[2,1],
                       height_ratios=[1,1]
                       )
    
    ax = plt.subplot(gs[0])                   
    # Create an axes instance
    # ax = fig.add_subplot(121)
    # plt.boxplot(data, sym='')
    ax.set_xticklabels(['A','B','C','D','A','B','C','D'])
    # ax.set_xticklabels(['B0639\n TSC','B0643\n TSC','B0645\n TSC','B0664\n TSC','B0639\n VFA','B0643\n VFA','B0645\n VFA','B0664\n VFA'])
    # ax.set_xticklabels(['B0639\n TSC','B0639\n VFA','B0643\n TSC','B0643\n VFA','B0645\n TSC','B0645\n VFA','B0664\n TSC','B0664\n VFA'])
    # bp = ax.boxplot(data, patch_artist=True, notch=True)
    bp = ax.boxplot(data, patch_artist=True)
    ax.plot(positions, means, 'ks')
    # ax[1].set_title("Plotting means manually")
    # print bp['boxes']
    # for box in bp['boxes']:
    for idx, box in enumerate(bp['boxes']):
        # print box
        # change outline color
        box.set( color='#222222', linewidth=2)
        # if idx%2==0:
            # box.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # box.set( color='#1C6047', linewidth=2)    
        # change fill color
        box.set( facecolor = '#FFFFFF' )
        
        
        
    # for cap in bp['caps']:
    i=0
    for idx, cap in enumerate(bp['caps']):
        cap.set(color='#222222', linewidth=2)
        # print(cap)
        # if i==4: 
            # i=0
        # if idx%2==0 and i<2:
            # cap.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # cap.set( color='#1C6047', linewidth=2)    
            # i=i+1
    for whisker in bp['whiskers']:
    # i=0
    # for idx, whisker in enumerate(bp['whiskers']):
        # print whisker
        # if i==8: 
            # i=0
        whisker.set(color='#222222', linewidth=2)
        # if idx%2==0  and i<2:
            # whisker.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # whisker.set( color='#1C6047', linewidth=2)    
            # i=i+1
            
    ## change color and linewidth of the medians
    for median in bp['medians']:
    # for idx, median in enumerate(bp['medians']):
        median.set(color='#222222', linewidth=2)
        # if idx%2==0:
            # median.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # median.set( color='#1C6047', linewidth=2)    

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#222222', alpha=0)
        
    # means = [np.mean(val1),np.mean(val2),np.mean(val3),np.mean(val4),np.mean(val5),np.mean(val6),np.mean(val7),np.mean(val8)]
    # ax[1].plot(range(8), means, 'rs')    
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    plt.title('[Na] in homogeneous WM')
    plt.ylabel('[Na] in mmol/L')
    plt.ylim(10, 70)
    maxbound=0.1
    T11=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0639_SandroDenoise/T1-3D_WM.nii",0,maxbound)
    T12=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0643_SandroDenoise/T1-3D_WM.nii",0,maxbound)
    T13=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0645_SandroDenoise/T1-3D_WM.nii",0,maxbound)
    T14=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0664_SandroDenoise/T1-3D_WM.nii",0,maxbound)
    
    data2 = [T11,T12,T13,T14]
    
    means2 = [np.mean(data12) for data12 in data2]
    positions2 = np.arange(4) + 1
    ax2 = plt.subplot(gs[1])
    # ax2 = fig.add_subplot(122)
    # plt.boxplot(data2, sym='')
    ax2.set_xticklabels(['A','B','C','D'])
    # ax.set_xticklabels(['B0639\n TSC','B0643\n TSC','B0645\n TSC','B0664\n TSC','B0639\n VFA','B0643\n VFA','B0645\n VFA','B0664\n VFA'])
    # ax.set_xticklabels(['B0639\n TSC','B0639\n VFA','B0643\n TSC','B0643\n VFA','B0645\n TSC','B0645\n VFA','B0664\n TSC','B0664\n VFA'])
    # bp = ax2.boxplot(data2, patch_artist=True, notch=True)
    bp = ax2.boxplot(data2, patch_artist=True)
    ax2.plot(positions2, means2, 'ks')
    # print bp['boxes']
    # for box in bp['boxes']:
    for idx, box in enumerate(bp['boxes']):
        # print box
        # change outline color
        box.set( color='#222222', linewidth=2)
        # if idx%2==0:
            # box.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # box.set( color='#1C6047', linewidth=2)    
        # change fill color
        box.set( facecolor = '#FFFFFF' )
        
        
        
    # for cap in bp['caps']:
    i=0
    for idx, cap in enumerate(bp['caps']):
        cap.set(color='#222222', linewidth=2)
        # print(cap)
        # if i==4: 
            # i=0
        # if idx%2==0 and i<2:
            # cap.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # cap.set( color='#1C6047', linewidth=2)    
            # i=i+1
    for whisker in bp['whiskers']:
    # i=0
    # for idx, whisker in enumerate(bp['whiskers']):
        # print whisker
        # if i==8: 
            # i=0
        whisker.set(color='#222222', linewidth=2)
        # if idx%2==0  and i<2:
            # whisker.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # whisker.set( color='#1C6047', linewidth=2)    
            # i=i+1
            
    ## change color and linewidth of the medians
    for median in bp['medians']:
    # for idx, median in enumerate(bp['medians']):
        median.set(color='#222222', linewidth=2)
        # if idx%2==0:
            # median.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # median.set( color='#1C6047', linewidth=2)    

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#222222', alpha=0)
        
    # means = [np.mean(val1),np.mean(val2),np.mean(val3),np.mean(val4),np.mean(val5),np.mean(val6),np.mean(val7),np.mean(val8)]
    # ax[1].plot(range(8), means, 'rs')    
    ## Remove top axes and right axes ticks
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_left()
    
    plt.title('T1 in homogeneous WM')
    plt.ylabel('T1 in s')
    plt.ylim(0, 0.07)
    
    plt.show()    
    
def BoxPlotCompareDistrib2():
    
    import nibabel as nib
    from scipy import stats
    from matplotlib import gridspec
    maxbound=120
        
    from processingFunctions import MeanStdCalc2
    TSCT1uncorrected100=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC100_NoT1_Concentration_central.nii",0,maxbound)
    TSCT1uncorrected120=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC120_NoT1_Concentration_central.nii",0,maxbound)
    TSCT1uncorrected150=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC150_NoT1_Concentration_central.nii",0,maxbound)
    TSCT1uncorrected200=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC200_NoT1_Concentration_central.nii",0,maxbound)
    TSCT1uncorrected300=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC300_NoT1_Concentration_central.nii",0,maxbound)
    
    TSCT1corrected100=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC100_T1VFA_Concentration_central.nii",0,maxbound)
    TSCT1corrected120=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC120_T1VFA_Concentration_central.nii",0,maxbound)
    TSCT1corrected150=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC150_T1VFA_Concentration_central.nii",0,maxbound)
    TSCT1corrected200=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC200_T1VFA_Concentration_central.nii",0,maxbound)
    TSCT1corrected300=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC300_T1VFA_Concentration_central.nii",0,maxbound)
    
    VFA=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/VFA_Concentration_central.nii",0,maxbound)
    import matplotlib.pyplot as plt
    data = [TSCT1uncorrected100,TSCT1corrected100,TSCT1uncorrected120,TSCT1corrected120,TSCT1uncorrected150,TSCT1corrected150,TSCT1uncorrected200,TSCT1corrected200,TSCT1uncorrected300,TSCT1corrected300,VFA]
    # data = [val1,val2,val3,val4,val5,val6,val7,val8]
    # labels = list('B0639','B0643','B0645','B0664')
    # multiple box plots on one figure
    fig = plt.figure(1)

    means = [np.mean(data1) for data1 in data]
    length = [np.size(data1) for data1 in data]
    # SDOM = [means[l]/np.sqrt(length[l]) for l in len(length)]
    print(means)
    print(length)
    positions = np.arange(11) + 1
    gs = gridspec.GridSpec(1, 2,
                       width_ratios=[2,0.5],
                       height_ratios=[1,1]
                       )
    
    ax = plt.subplot(gs[0])                   
    # Create an axes instance
    # ax = fig.add_subplot(121)
    # plt.boxplot(data, sym='')
    # ax.set_xticklabels(['NUFFT','FISTA','NUFFT','FISTA'])
    ax.set_xticklabels(['TR100','TR100','TR120','TR120','TR150','TR150','TR200','TR200','TR300', 'TR300','VFA'])
    # ax.set_xticklabels(['B0639\n TSC','B0643\n TSC','B0645\n TSC','B0664\n TSC','B0639\n VFA','B0643\n VFA','B0645\n VFA','B0664\n VFA'])
    # ax.set_xticklabels(['B0639\n TSC','B0639\n VFA','B0643\n TSC','B0643\n VFA','B0645\n TSC','B0645\n VFA','B0664\n TSC','B0664\n VFA'])
    # bp = ax.boxplot(data, patch_artist=True, notch=True)
    bp = ax.boxplot(data, patch_artist=True)
    ax.plot(positions, means, 'ks')
    # ax[1].set_title("Plotting means manually")
    # print bp['boxes']
    # for box in bp['boxes']:
    for idx, box in enumerate(bp['boxes']):
        # print box
        # change outline color
        box.set( color='#222222', linewidth=2)
        # if idx%2==0:
            # box.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # box.set( color='#1C6047', linewidth=2)    
        # change fill color
        box.set( facecolor = '#FFFFFF' )
        
        
        
    # for cap in bp['caps']:
    i=0
    for idx, cap in enumerate(bp['caps']):
        cap.set(color='#222222', linewidth=2)
        # print(cap)
        # if i==4: 
            # i=0
        # if idx%2==0 and i<2:
            # cap.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # cap.set( color='#1C6047', linewidth=2)    
            # i=i+1
    for whisker in bp['whiskers']:
    # i=0
    # for idx, whisker in enumerate(bp['whiskers']):
        # print whisker
        # if i==8: 
            # i=0
        whisker.set(color='#222222', linewidth=2)
        # if idx%2==0  and i<2:
            # whisker.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # whisker.set( color='#1C6047', linewidth=2)    
            # i=i+1
            
    ## change color and linewidth of the medians
    for median in bp['medians']:
    # for idx, median in enumerate(bp['medians']):
        median.set(color='#222222', linewidth=2)
        # if idx%2==0:
            # median.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # median.set( color='#1C6047', linewidth=2)    

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#222222', alpha=0)
        
    # means = [np.mean(val1),np.mean(val2),np.mean(val3),np.mean(val4),np.mean(val5),np.mean(val6),np.mean(val7),np.mean(val8)]
    # ax[1].plot(range(8), means, 'rs')    
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    plt.title('[Na] in Liquid Compartment')
    plt.ylabel('[Na] in mmol/L')
    plt.ylim(50, 90)
    
    maxbound=0.1
    T11=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/den_T1-3D_central.nii",0,maxbound)
    # T12=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/RecoComparaison/FISTA_T1_image_WM.nii",0,maxbound)
    
    data2 = [T11]
    
    means2 = [np.mean(data12) for data12 in data2]
    positions2 = np.arange(1) + 1
    ax2 = plt.subplot(gs[1])
    # ax2 = fig.add_subplot(122)
    # plt.boxplot(data2, sym='')
    ax2.set_xticklabels(['VFA'])
    # ax.set_xticklabels(['B0639\n TSC','B0643\n TSC','B0645\n TSC','B0664\n TSC','B0639\n VFA','B0643\n VFA','B0645\n VFA','B0664\n VFA'])
    # ax.set_xticklabels(['B0639\n TSC','B0639\n VFA','B0643\n TSC','B0643\n VFA','B0645\n TSC','B0645\n VFA','B0664\n TSC','B0664\n VFA'])
    # bp = ax2.boxplot(data2, patch_artist=True, notch=True)
    bp = ax2.boxplot(data2, patch_artist=True)
    ax2.plot(positions2, means2, 'ks')
    # print bp['boxes']
    # for box in bp['boxes']:
    for idx, box in enumerate(bp['boxes']):
        # print box
        # change outline color
        box.set( color='#222222', linewidth=2)
        # if idx%2==0:
            # box.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # box.set( color='#1C6047', linewidth=2)    
        # change fill color
        box.set( facecolor = '#FFFFFF' )
        
        
        
    # for cap in bp['caps']:
    i=0
    for idx, cap in enumerate(bp['caps']):
        cap.set(color='#222222', linewidth=2)
        # print(cap)
        # if i==4: 
            # i=0
        # if idx%2==0 and i<2:
            # cap.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # cap.set( color='#1C6047', linewidth=2)    
            # i=i+1
    for whisker in bp['whiskers']:
    # i=0
    # for idx, whisker in enumerate(bp['whiskers']):
        # print whisker
        # if i==8: 
            # i=0
        whisker.set(color='#222222', linewidth=2)
        # if idx%2==0  and i<2:
            # whisker.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # whisker.set( color='#1C6047', linewidth=2)    
            # i=i+1
            
    ## change color and linewidth of the medians
    for median in bp['medians']:
    # for idx, median in enumerate(bp['medians']):
        median.set(color='#222222', linewidth=2)
        # if idx%2==0:
            # median.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # median.set( color='#1C6047', linewidth=2)    

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#222222', alpha=0)
        
    # means = [np.mean(val1),np.mean(val2),np.mean(val3),np.mean(val4),np.mean(val5),np.mean(val6),np.mean(val7),np.mean(val8)]
    # ax[1].plot(range(8), means, 'rs')    
    ## Remove top axes and right axes ticks
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_left()
    
    plt.title('T1 in Liquid Compartment')
    plt.ylabel('T1 in s')
    plt.ylim(0.02, 0.06)
    
    plt.show()    
    
    
def BoxPlotCompareDistrib2plots():
    
    import nibabel as nib
    from scipy import stats
    from matplotlib import gridspec
    maxbound=150
        
    from processingFunctions import MeanStdCalc2
    val1=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/Phantom_3VFA_6TSC-SandroDen/ReprocessingMarch17/CorrectedTSC_withB1sensibility_andT1map_Tube2%haut_5slicesCentralROI.nii",0,maxbound)
    val2=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/Phantom_3VFA_6TSC-SandroDen/ReprocessingMarch17/VFA_LargeTubeROIcalib_CentralROI.nii",0,maxbound)

    import matplotlib.pyplot as plt
    data = [val1,val2]
    # data = [val1,val2,val3,val4,val5,val6,val7,val8]
    # labels = list('B0639','B0643','B0645','B0664')
    # multiple box plots on one figure
    fig = plt.figure(1)
    
    means = [np.mean(data1) for data1 in data]
    length = [np.size(data1) for data1 in data]
    # SDOM = [means[l]/np.sqrt(length[l]) for l in len(length)]
    print(means)
    print(length)
    positions = np.arange(2) + 1
    gs = gridspec.GridSpec(1, 1,
                       width_ratios=[2,1],
                       height_ratios=[2,1]
                       )
    
    ax = plt.subplot(gs[0])                   
    # Create an axes instance
    # ax = fig.add_subplot(121)
    # plt.boxplot(data, sym='')
    ax.set_xticklabels(['TSC','VFA'])
    # ax.set_xticklabels(['B0639\n TSC','B0643\n TSC','B0645\n TSC','B0664\n TSC','B0639\n VFA','B0643\n VFA','B0645\n VFA','B0664\n VFA'])
    # ax.set_xticklabels(['B0639\n TSC','B0639\n VFA','B0643\n TSC','B0643\n VFA','B0645\n TSC','B0645\n VFA','B0664\n TSC','B0664\n VFA'])
    # bp = ax.boxplot(data, patch_artist=True, notch=True)
    bp = ax.boxplot(data, patch_artist=True)
    ax.plot(positions, means, 'ks')
    # ax[1].set_title("Plotting means manually")
    # print bp['boxes']
    # for box in bp['boxes']:
    for idx, box in enumerate(bp['boxes']):
        # print box
        # change outline color
        box.set( color='#222222', linewidth=2)
        # if idx%2==0:
            # box.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # box.set( color='#1C6047', linewidth=2)    
        # change fill color
        box.set( facecolor = '#FFFFFF' )
        
        
        
    # for cap in bp['caps']:
    i=0
    for idx, cap in enumerate(bp['caps']):
        cap.set(color='#222222', linewidth=2)
        # print(cap)
        # if i==4: 
            # i=0
        # if idx%2==0 and i<2:
            # cap.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # cap.set( color='#1C6047', linewidth=2)    
            # i=i+1
    for whisker in bp['whiskers']:
    # i=0
    # for idx, whisker in enumerate(bp['whiskers']):
        # print whisker
        # if i==8: 
            # i=0
        whisker.set(color='#222222', linewidth=2)
        # if idx%2==0  and i<2:
            # whisker.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # whisker.set( color='#1C6047', linewidth=2)    
            # i=i+1
            
    ## change color and linewidth of the medians
    for median in bp['medians']:
    # for idx, median in enumerate(bp['medians']):
        median.set(color='#222222', linewidth=2)
        # if idx%2==0:
            # median.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # median.set( color='#1C6047', linewidth=2)    

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#222222', alpha=0)
        
    # means = [np.mean(val1),np.mean(val2),np.mean(val3),np.mean(val4),np.mean(val5),np.mean(val6),np.mean(val7),np.mean(val8)]
    # ax[1].plot(range(8), means, 'rs')    
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    plt.title('[Na] in Central ROI')
    plt.ylabel('[Na] in mmol/L')
    plt.ylim(50, 100)    
    
    plt.show()
    
def AffineCalibrationMultiPoints(Im1,Im2):

    import nibabel as nib
    import numpy, os, sys
    
    # Im1 = nib.load(Im1)
    # Im1 =Im1.get_data()
    # Im2 = nib.load(Im2)
    # Im2 =Im2.get_data()    
    
    from processingFunctions import MeanStdCalc2
    val1=MeanStdCalc2(Im1,0,120)
    val2=MeanStdCalc2(Im2,0,150)
    print(len(val1), len(val2))
    x1=np.zeros(len(val1))
    x2=np.zeros(len(val2))
    
    from scipy import stats
    y = np.zeros(len(val1)+len(val2))
    x = np.zeros(len(val1)+len(val2))
    print(len(x),len(y))
    x2[:]=100
    x1[:]=50
    
    x=np.concatenate([x1,x2])
    y=np.concatenate([val1,val2])
    print(len(x),len(y))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y);
    print(slope, intercept)

    
def CorrectTSCImagewithT1map(TSC,B1map,T1map,TR):

    import nibabel as nib
    import numpy, os, sys
    
    B1map = nib.load(B1map)
    B1map =B1map.get_data()
    T1map = nib.load(T1map)
    T1map =T1map.get_data()
    TSC = nib.load(TSC)
    TSC =TSC.get_data()
    # TR=0.120;
    CorrectedTSC= numpy.zeros(shape=TSC.shape)
    
    Alpha=3.1415926535/2;
    
    for i in range(TSC.shape[0]):
        for j in range(TSC.shape[1]):
            for k in range(TSC.shape[2]):
                E1=numpy.exp(-TR/T1map[i,j,k])
                CorrectedTSC[i,j,k]=TSC[i,j,k]*((1-(numpy.cos(B1map[i,j,k]*Alpha)*E1))/(1-E1))
                CorrectedTSC[i,j,k]=CorrectedTSC[i,j,k]/B1map[i,j,k]        # Correction for B1-
    
    from nifty_funclib import SaveArrayAsNIfTI
    # Hpath, Fname = os.path.split(str(TSC))
    # print Hpath
    # Fname = Fname.split('.')
    # OutputPath = os.path.join( Hpath + '\\' + "CorrectedTSC.nii")
    # print OutputPath
    SaveArrayAsNIfTI(CorrectedTSC,4,4,4,"C:/Users/AC243636/Pictures/CorrectedTSC_withB1sensibility_andT1map.nii")    
        
        
def BoxPlotCompareDistribTSCVFA():
    
    import nibabel as nib
    from scipy import stats
    from matplotlib import gridspec
    maxbound=120
        
    from processingFunctions import MeanStdCalc2
    # TSCT1uncorrected100=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC100_NoT1_Concentration_central.nii",0,maxbound)
    # TSCT1uncorrected120=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC120_NoT1_Concentration_central.nii",0,maxbound)
    # TSCT1uncorrected150=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC150_NoT1_Concentration_central.nii",0,maxbound)
    # TSCT1uncorrected200=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC200_NoT1_Concentration_central.nii",0,maxbound)
    # TSCT1uncorrected300=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC300_NoT1_Concentration_central.nii",0,maxbound)
    
    # TSCT1uncorrected100=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC100-Tubes6_NoT1_Concentration_central.nii",0,maxbound)
    # TSCT1uncorrected120=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC120-Tubes6_NoT1_Concentration_central.nii",0,maxbound)
    # TSCT1uncorrected150=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC150-Tubes6_NoT1_Concentration_central.nii",0,maxbound)
    # TSCT1uncorrected200=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC200-Tubes6_NoT1_Concentration_central.nii",0,maxbound)
    # TSCT1uncorrected300=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC300-Tubes6_NoT1_Concentration_central.nii",0,maxbound)
    
    TSCT1uncorrected100=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC100_T1VFA_Concentration_central.nii",0,maxbound)
    TSCT1uncorrected120=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC120_T1VFA_Concentration_central.nii",0,maxbound)
    TSCT1uncorrected150=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC150_T1VFA_Concentration_central.nii",0,maxbound)
    TSCT1uncorrected200=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC200_T1VFA_Concentration_central.nii",0,maxbound)
    TSCT1uncorrected300=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC300_T1VFA_Concentration_central.nii",0,maxbound)
    
    TSCT1corrected100=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC100-Tubes6_T1VFA_Concentration_central.nii",0,maxbound)
    TSCT1corrected120=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC120-Tubes6_T1VFA_Concentration_central.nii",0,maxbound)
    TSCT1corrected150=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC150-Tubes6_T1VFA_Concentration_central.nii",0,maxbound)
    TSCT1corrected200=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC200-Tubes6_T1VFA_Concentration_central.nii",0,maxbound)
    TSCT1corrected300=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC300-Tubes6_T1VFA_Concentration_central.nii",0,maxbound)
    
    # TSCT1uncorrected100=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC100-Tubes6_NoT1_Concentration_central.nii",0,maxbound)
    # TSCT1uncorrected120=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC120-Tubes6_NoT1_Concentration_central.nii",0,maxbound)
    # TSCT1uncorrected150=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC150-Tubes6_NoT1_Concentration_central.nii",0,maxbound)
    # TSCT1uncorrected200=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC200-Tubes6_NoT1_Concentration_central.nii",0,maxbound)
    # TSCT1uncorrected300=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC300-Tubes6_NoT1_Concentration_central.nii",0,maxbound)
    
    
    VFA=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/VFA_Concentration_central.nii",0,maxbound)
    VFA6=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/VFA-Tubes6_Concentration_central.nii",0,maxbound)
    
    import matplotlib.pyplot as plt
    data = [TSCT1uncorrected100,TSCT1corrected100,TSCT1uncorrected120,TSCT1corrected120,TSCT1uncorrected150,TSCT1corrected150,TSCT1uncorrected200,TSCT1corrected200,TSCT1uncorrected300,TSCT1corrected300,VFA,VFA6]
    # data = [TSCT1corrected100,TSCT1uncorrected100,TSCT1corrected120,TSCT1uncorrected120,TSCT1corrected150,TSCT1uncorrected150,TSCT1corrected200,TSCT1uncorrected200,TSCT1corrected300,TSCT1uncorrected300,VFA,VFA6]
    # data = [val1,val2,val3,val4,val5,val6,val7,val8]
    # labels = list('B0639','B0643','B0645','B0664')
    # multiple box plots on one figure
    fig = plt.figure(1)

    means = [np.mean(data1) for data1 in data]
    length = [np.size(data1) for data1 in data]
    # SDOM = [means[l]/np.sqrt(length[l]) for l in len(length)]
    errors = [(75-m)/75*100 for m in means]
    print(means)
    print(length)
    print(errors)
    positions = np.arange(12) + 1
    gs = gridspec.GridSpec(1, 1,
                       width_ratios=[2],
                       height_ratios=[1]
                       )
    
    # ax = plt.subplot(gs[0,0])                   
    ax = plt.subplot(gs[0])                   
    # Create an axes instance
    # ax = fig.add_subplot(121)
    # plt.boxplot(data, sym='')
    # ax.set_xticklabels(['NUFFT','FISTA','NUFFT','FISTA'])
    # ax.set_xticklabels(['TR100 \n T1 U','TR100 \n T1 C','TR120 \n T1 U','TR120 \n T1 C','TR150 \n T1 U','TR150 \n T1 C','TR200 \n T1 U','TR200 \n T1 C','TR300 \n T1 U', 'TR300 \n T1 C','VFA'])
    ax.set_xticklabels(['T1 Uncorrected 2%','T1 Uncorrected 6%','T1 Uncorrected','T1 Corrected','T1 Uncorrected','T1 Corrected','T1 Uncorrected','T1 Corrected','T1 Uncorrected', 'T1 Corrected','VFA'])
    # ax.set_xticklabels(['B0639\n TSC','B0643\n TSC','B0645\n TSC','B0664\n TSC','B0639\n VFA','B0643\n VFA','B0645\n VFA','B0664\n VFA'])
    # ax.set_xticklabels(['B0639\n TSC','B0639\n VFA','B0643\n TSC','B0643\n VFA','B0645\n TSC','B0645\n VFA','B0664\n TSC','B0664\n VFA'])
    # bp = ax.boxplot(data, patch_artist=True, notch=True)
    bp = ax.boxplot(data, patch_artist=True)
    ax.plot(positions, means, 'ks')
    # ax[1].set_title("Plotting means manually")
    # print bp['boxes']
    # for box in bp['boxes']:
    for idx, box in enumerate(bp['boxes']):
        # print box
        # change outline color
        box.set( color='#222222', linewidth=2)
        # if idx%2==0:
            # box.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # box.set( color='#1C6047', linewidth=2)    
        # change fill color
        box.set( facecolor = '#FFFFFF' )
        
        
        
    # for cap in bp['caps']:
    i=0
    for idx, cap in enumerate(bp['caps']):
        cap.set(color='#222222', linewidth=2)
        # print(cap)
        # if i==4: 
            # i=0
        # if idx%2==0 and i<2:
            # cap.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # cap.set( color='#1C6047', linewidth=2)    
            # i=i+1
    for whisker in bp['whiskers']:
    # i=0
    # for idx, whisker in enumerate(bp['whiskers']):
        # print whisker
        # if i==8: 
            # i=0
        whisker.set(color='#222222', linewidth=2)
        # if idx%2==0  and i<2:
            # whisker.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # whisker.set( color='#1C6047', linewidth=2)    
            # i=i+1
            
    ## change color and linewidth of the medians
    for median in bp['medians']:
    # for idx, median in enumerate(bp['medians']):
        median.set(color='#222222', linewidth=2)
        # if idx%2==0:
            # median.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # median.set( color='#1C6047', linewidth=2)    

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#222222', alpha=0)
        
    # means = [np.mean(val1),np.mean(val2),np.mean(val3),np.mean(val4),np.mean(val5),np.mean(val6),np.mean(val7),np.mean(val8)]
    # ax[1].plot(range(8), means, 'rs')    
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    plt.title('Influence of T1 on measured Sodium Concentration')
    plt.ylabel('[Na] in mmol/L')
    plt.ylim(50, 100)
    
    plt.axhline(y=75, xmin=0, xmax=4, linewidth=2,linestyle='--', color = 'k')
    for i in range(len(positions)):
        plt.text(i+1, 52, '%.1f' % errors[i],horizontalalignment='center', size=12)
        if ((i+1)%2)==0:
            plt.axvline(x=i+1+0.5, linewidth=2, color = 'k')
        

    
    maxbound=0.1
    T11=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/den_T1-3D_central.nii",0,maxbound)
    # T12=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/RecoComparaison/FISTA_T1_image_WM.nii",0,maxbound)
    
    # data2 = [T11]
    
    # means2 = [np.mean(data12) for data12 in data2]
    # positions2 = np.arange(1) + 1
    # ax2 = plt.subplot(gs[:,1])
    # ax2 = plt.subplot(gs[1])
    # ax2 = fig.add_subplot(122)
    # plt.boxplot(data2, sym='')
    # ax2.set_xticklabels(['VFA'])
    # ax.set_xticklabels(['B0639\n TSC','B0643\n TSC','B0645\n TSC','B0664\n TSC','B0639\n VFA','B0643\n VFA','B0645\n VFA','B0664\n VFA'])
    # ax.set_xticklabels(['B0639\n TSC','B0639\n VFA','B0643\n TSC','B0643\n VFA','B0645\n TSC','B0645\n VFA','B0664\n TSC','B0664\n VFA'])
    # bp = ax2.boxplot(data2, patch_artist=True, notch=True)
    # bp = ax2.boxplot(data2, patch_artist=True)
    # ax2.plot(positions2, means2, 'ks')
    # print bp['boxes']
    # for box in bp['boxes']:
    # for idx, box in enumerate(bp['boxes']):
        # print box
        # change outline color
        # box.set( color='#222222', linewidth=2)
        # if idx%2==0:
            # box.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # box.set( color='#1C6047', linewidth=2)    
        # change fill color
        # box.set( facecolor = '#FFFFFF' )
        
        
        
    # for cap in bp['caps']:
    # i=0
    # for idx, cap in enumerate(bp['caps']):
        # cap.set(color='#222222', linewidth=2)
        # print(cap)
        # if i==4: 
            # i=0
        # if idx%2==0 and i<2:
            # cap.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # cap.set( color='#1C6047', linewidth=2)    
            # i=i+1
    # for whisker in bp['whiskers']:
    # i=0
    # for idx, whisker in enumerate(bp['whiskers']):
        # print whisker
        # if i==8: 
            # i=0
        # whisker.set(color='#222222', linewidth=2)
        # if idx%2==0  and i<2:
            # whisker.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # whisker.set( color='#1C6047', linewidth=2)    
            # i=i+1
            
    ## change color and linewidth of the medians
    # for median in bp['medians']:
    # for idx, median in enumerate(bp['medians']):
        # median.set(color='#222222', linewidth=2)
        # if idx%2==0:
            # median.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # median.set( color='#1C6047', linewidth=2)    

    ## change the style of fliers and their fill
    # for flier in bp['fliers']:
        # flier.set(marker='o', color='#222222', alpha=0)
        
    # means = [np.mean(val1),np.mean(val2),np.mean(val3),np.mean(val4),np.mean(val5),np.mean(val6),np.mean(val7),np.mean(val8)]
    # ax[1].plot(range(8), means, 'rs')    
    ## Remove top axes and right axes ticks
    # ax2.get_xaxis().tick_bottom()
    # ax2.get_yaxis().tick_left()
    # plt.title('T1 in Liquid Compartment')
    # plt.ylabel('T1 in s')
    # plt.ylim(0.02, 0.06)
    # pos = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5]
    # ax3= plt.subplot(gs[1,0])
    # bar=ax3.bar(pos,errors, alpha=0.75, color='black',)
    
    # ax3.set_xticklabels(['TR100','TR100','TR120','TR120','TR150','TR150','TR200','TR200','TR300', 'TR300','VFA'])
    # plt.title('Error in measurements')
    # plt.ylabel('%')
    
    plt.show()    

def BoxPlotCompareDistribTSCVFAInVivo():
    
    import nibabel as nib
    from scipy import stats
    from matplotlib import gridspec
    maxbound=120
        
    from processingFunctions import MeanStdCalc2
    B0639NoT1=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0639_SandroDenoise/NewReprocessing/NaNoNoise_CorrectedTSC_withB1sensibilityNoT1_WM.nii",0,maxbound)
    # B0639T1=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0639_SandroDenoise/NewReprocessing/NaNoNoise_CorrectedTSC_withB1sensibility_andT1map_WM.nii",0,maxbound)
    B0639VFA=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0639_SandroDenoise/NewReprocessing/NaNoNoise_SpinDensity_WM.nii",0,maxbound)
    
    B0643NoT1=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0643_SandroDenoise/NewReprocessing/NaNoNoise_CorrectedTSC_withB1sensibilityNoT1_WM.nii",0,maxbound)
    # B0643T1=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0643_SandroDenoise/NewReprocessing/NaNoNoise_CorrectedTSC_withB1sensibility_andT1map_WM.nii",0,maxbound)
    B0643VFA=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0643_SandroDenoise/NewReprocessing/NaNoNoise_SpinDensity_WM.nii",0,maxbound)
    
    B0645NoT1=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0645_SandroDenoise/NewReprocessing/NaNoNoise_CorrectedTSC_withB1sensibilityNoT1_WM.nii",0,maxbound)
    # B0645T1=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0645_SandroDenoise/NewReprocessing/NaNoNoise_CorrectedTSC_withB1sensibility_andT1map_WM.nii",0,maxbound)
    B0645VFA=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0645_SandroDenoise/NewReprocessing/NaNoNoise_SpinDensity_WM.nii",0,maxbound)
    
    B0664NoT1=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0664_SandroDenoise/NewReprocessing/NaNoNoise_CorrectedTSC_withB1sensibilityNoT1_WM.nii",0,maxbound)
    # B0664T1=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0664_SandroDenoise/NewReprocessing/NaNoNoise_CorrectedTSC_withB1sensibility_andT1map_WM.nii",0,maxbound)
    B0664VFA=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0664_SandroDenoise/NewReprocessing/NaNoNoise_SpinDensity_WM.nii",0,maxbound)
    
    import matplotlib.pyplot as plt
    # data = [B0639NoT1,B0639T1,B0639VFA,B0643NoT1,B0643T1,B0643VFA,B0645NoT1,B0645T1,B0645VFA,B0664NoT1,B0664T1,B0664VFA]
    data = [B0639NoT1,B0639VFA,B0643NoT1,B0643VFA,B0645NoT1,B0645VFA,B0664NoT1,B0664VFA]
    # data = [val1,val2,val3,val4,val5,val6,val7,val8]
    # labels = list('B0639','B0643','B0645','B0664')
    # multiple box plots on one figure
    fig = plt.figure(1)
    
    font = {'family' : 'normal','weight' : 'bold','size'   : 20}

    plt.rc('font', **font)

    means = [np.mean(data1) for data1 in data]
    # length = [np.size(data1) for data1 in data]
    # SDOM = [means[l]/np.sqrt(length[l]) for l in len(length)]
    # errors = [(75-m)/75*100 for m in means]
    print(means)
    # print length
    # print errors
    # positions = np.arange(8) + 1
    positions = [1,2,4,5,7,8,10,11]
    gs = gridspec.GridSpec(1, 2,
                       width_ratios=[2,1],
                       height_ratios=[1,0]
                       )
    
    # ax = plt.subplot(gs[0,0])                   
    ax = plt.subplot(gs[0])               
    # Create an axes instance
    # ax = fig.add_subplot(121)
    # plt.boxplot(data, sym='')
    # ax.set_xticklabels(['NUFFT','FISTA','NUFFT','FISTA'])
    # ax.set_xticklabels(['B0639NoT1','B0639T1','B0639VFA','B0643NoT1','B0643T1','B0643VFA','B0645NoT1','B0645T1','B0645VFA','B0664NoT1','B0664T1','B0664VFA'])
    # ax.set_xticklabels(['B0639NoT1','B0639VFA','B0643NoT1','B0643VFA','B0645NoT1','B0645VFA','B0664NoT1','B0664VFA'])
    ax.set_xticklabels(['TSC \n Subject 1','VFA \n Subject 1','TSC \n Subject 2','VFA \n Subject 2','TSC \n Subject 3','VFA \n Subject 3','TSC \n Subject 4','VFA \n Subject 4'])
    # ax.set_xticklabels(['B0639\n TSC','B0643\n TSC','B0645\n TSC','B0664\n TSC','B0639\n VFA','B0643\n VFA','B0645\n VFA','B0664\n VFA'])
    # ax.set_xticklabels(['B0639\n TSC','B0639\n VFA','B0643\n TSC','B0643\n VFA','B0645\n TSC','B0645\n VFA','B0664\n TSC','B0664\n VFA'])
    # bp = ax.boxplot(data, patch_artist=True, notch=True)
    bp = ax.boxplot(data,positions =positions, patch_artist=True)
    ax.plot(positions, means, 'ks')
    # ax[1].set_title("Plotting means manually")
    # print bp['boxes']
    # for box in bp['boxes']:
    for idx, box in enumerate(bp['boxes']):
        # print box
        # change outline color
        if idx%2==0:
            box.set( color='#222222', linewidth=3)
        # if idx%2==0:
            # box.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # box.set( color='#1C6047', linewidth=2)    
        # change fill color
            box.set( facecolor = '#FFFFFF' )
        else:
            box.set( color='#3B5998', linewidth=3)
            box.set( facecolor = '#FFFFFF' )
        
    # for cap in bp['caps']:
    i=0
    for idx, cap in enumerate(bp['caps']):
        # if idx%2==0:
        cap.set(color='#222222', linewidth=3)
        # else:    
            # cap.set(color='#3B5998', linewidth=3)
        # print(cap)
        # if i==4: 
            # i=0
        # if idx%2==0 and i<2:
            # cap.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # cap.set( color='#1C6047', linewidth=2)    
            # i=i+1
    for idx, whisker in enumerate(bp['whiskers']):
        # if idx%2==0:
        whisker.set(color='#222222', linewidth=3)
        # else:    
            # whisker.set(color='#3B5998', linewidth=2)
        
        
        # if idx%2==0  and i<2:
            # whisker.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # whisker.set( color='#1C6047', linewidth=2)    
            # i=i+1
            
    ## change color and linewidth of the medians
    for idx, median in enumerate(bp['medians']):
        if idx%2==0:
            median.set(color='#222222', linewidth=3)
        else:    
            median.set(color='#3B5998', linewidth=3)
        
        # if idx%2==0:
            # median.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # median.set( color='#1C6047', linewidth=2)    

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#222222', alpha=0)
        
    # means = [np.mean(val1),np.mean(val2),np.mean(val3),np.mean(val4),np.mean(val5),np.mean(val6),np.mean(val7),np.mean(val8)]
    # ax[1].plot(range(8), means, 'rs')    
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # import matplotlib.patches as mpatches
    # red_patch = mpatches.Patch(color='red', label='TSC')
    # blue_patch = mpatches.Patch(color='blue', label='VFA')
    # plt.legend(handles=[red_patch,blue_patch])
    plt.title('[Na] in Homogeneous WM')
    plt.ylabel('[Na] in mmol/L')
    plt.ylim(20, 60)
    
    # for i in range(len(positions)):
        # plt.text(i+1, 52, '%.1f' % errors[i],horizontalalignment='center', size=12)
        # if ((i+1)%2)==0:
            # plt.axvline(x=i+1+0.5, linewidth=2, color = 'k')
    
    # T12=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/RecoComparaison/FISTA_T1_image_WM.nii",0,maxbound)
    maxbound=0.1
    B0639T1=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0639_SandroDenoise/T1-3D_WM.nii",0,maxbound)
    B0643T1=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0643_SandroDenoise/T1-3D_WM.nii",0,maxbound)
    B0645T1=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0645_SandroDenoise/T1-3D_WM.nii",0,maxbound)
    B0664T1=MeanStdCalc2("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/B0664_SandroDenoise/T1-3D_WM.nii",0,maxbound)
    
    data2 = [B0639T1,B0643T1,B0645T1,B0664T1]
    
    means2 = [np.mean(data12) for data12 in data2]
    positions2 = np.arange(4) + 1
    # ax2 = plt.subplot(gs[:,1])
    ax2 = plt.subplot(gs[1])
    # ax2 = fig.add_subplot(122)
    # plt.boxplot(data2, sym='')
    # ax2.set_xticklabels(['B0639','B0643','B0645','B0664'])
    ax2.set_xticklabels(['Subject 1','Subject 2','Subject 3','Subject 4'])
    # ax2.set_xticklabels(['VFA'])
    # ax.set_xticklabels(['B0639\n TSC','B0643\n TSC','B0645\n TSC','B0664\n TSC','B0639\n VFA','B0643\n VFA','B0645\n VFA','B0664\n VFA'])
    # ax.set_xticklabels(['B0639\n TSC','B0639\n VFA','B0643\n TSC','B0643\n VFA','B0645\n TSC','B0645\n VFA','B0664\n TSC','B0664\n VFA'])
    # bp = ax2.boxplot(data2, patch_artist=True, notch=True)
    bp = ax2.boxplot(data2, patch_artist=True)
    ax2.plot(positions2, means2, 'ks')
    # print bp['boxes']
    # for box in bp['boxes']:
    for idx, box in enumerate(bp['boxes']):
        # print box
        # change outline color
        box.set( color='#222222', linewidth=3)
        # if idx%2==0:
            # box.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # box.set( color='#1C6047', linewidth=2)    
        # change fill color
        box.set( facecolor = '#FFFFFF' )
        
        
        
    # for cap in bp['caps']:
    # i=0
    for idx, cap in enumerate(bp['caps']):
        cap.set(color='#222222', linewidth=3)
        # print(cap)
        # if i==4: 
            # i=0
        # if idx%2==0 and i<2:
            # cap.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # cap.set( color='#1C6047', linewidth=2)    
            # i=i+1
    # for whisker in bp['whiskers']:
    # i=0
    for idx, whisker in enumerate(bp['whiskers']):
        # print whisker
        # if i==8: 
            # i=0
        whisker.set(color='#222222', linewidth=3)
        # if idx%2==0  and i<2:
            # whisker.set( color='#C43210', linewidth=2)
            # i=i+1
        # if idx%2==1  and i>=2:
            # whisker.set( color='#1C6047', linewidth=2)    
            # i=i+1
            
    ## change color and linewidth of the medians
    # for median in bp['medians']:
    for idx, median in enumerate(bp['medians']):
        median.set(color='#222222', linewidth=3)
        # if idx%2==0:
            # median.set( color='#C43210', linewidth=2)
        # if idx%2==1:
            # median.set( color='#1C6047', linewidth=2)    

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#222222', alpha=0)
        
    # means = [np.mean(val1),np.mean(val2),np.mean(val3),np.mean(val4),np.mean(val5),np.mean(val6),np.mean(val7),np.mean(val8)]
    # ax[1].plot(range(8), means, 'rs')    
    ## Remove top axes and right axes ticks
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_left()
    plt.title('T1 in Homogeneous WM')
    plt.ylabel('T1 in s')
    plt.ylim(0.0, 0.06)
    
    plt.show()    
    
def ProcessMeasureCSV(file):
    import csv, numpy
    with open(file, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        T1m=0;T1std=0;VFA=0;VFAstd=0;TSC100U=0;TSC100Ustd=0;TSC100C=0;TSC100Cstd=0;
        TSC120U=0;TSC120Ustd=0;TSC120C=0;TSC120Cstd=0;TSC150U=0;TSC150Ustd=0;TSC150C=0;TSC150Cstd=0;
        TSC200U=0;TSC200Ustd=0;TSC200C=0;TSC200Cstd=0;TSC300U=0;TSC300Ustd=0;TSC300C=0;TSC300Cstd=0;
        
        colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
        dotsize=18
        import matplotlib.pyplot as plt
        fig = plt.figure(1)
        for idx, row in enumerate(spamreader):
            # print idx
            a=row[0].split("\t")
            mean=a[3]
            std=a[4]
            
            if idx==1189:
                break
        
            if idx%12==0:
                T1m=T1m+float(mean)
                T1std=T1std+(float(std))**2
                
            if idx%12==1:
                VFA=VFA+float(mean)
                VFAstd=VFAstd+(float(std))**2
                # plt.plot(4.25, float(mean), marker='.', color=colors[0], markersize=dotsize,alpha=0.75,label="VFA")
                
            if idx%12==2:
                TSC100U=TSC100U+float(mean)
                TSC100Ustd=TSC100Ustd+(float(std))**2
                # plt.plot(0.5, float(mean), marker='.', color=colors[2], markersize=dotsize,alpha=0.75,label="TSC TR 100 ms")
                
            if idx%12==3:
                TSC100C=TSC100C+float(mean)
                TSC100Cstd=TSC100Cstd+(float(std))**2
                # plt.plot(0.75, float(mean), marker='.', color=colors[1], markersize=dotsize,alpha=0.75,label="TSC TR 100 ms")
                
            if idx%12==4:
                TSC120U=TSC120U+float(mean)
                TSC120Ustd=TSC120Ustd+(float(std))**2
                # plt.plot(1.25, float(mean), marker='.', color=colors[2], markersize=dotsize,alpha=0.75,label="TSC TR 120 ms")
                
            if idx%12==5:
                TSC120C=TSC120C+float(mean)
                TSC120Cstd=TSC120Cstd+(float(std))**2
                # plt.plot(1.5, float(mean), marker='.', color=colors[1], markersize=dotsize,alpha=0.75,label="TSC TR 120 ms")
                
            if idx%12==6:
                TSC150U=TSC150U+float(mean)
                TSC150Ustd=TSC150Ustd+(float(std))**2
                # plt.plot(2, float(mean), marker='.', color=colors[2], markersize=dotsize,alpha=0.75,label="TSC TR 150 ms")
                
            if idx%12==7:
                TSC150C=TSC150C+float(mean)
                TSC150Cstd=TSC150Cstd+(float(std))**2
                # plt.plot(2.25, float(mean), marker='.', color=colors[1], markersize=dotsize,alpha=0.75,label="TSC TR 150 ms")
                
            if idx%12==8:
                TSC200U=TSC200U+float(mean)
                TSC200Ustd=TSC200Ustd+(float(std))**2
                # plt.plot(2.75, float(mean), marker='.', color=colors[2], markersize=dotsize,alpha=0.75,label="TSC TR 200 ms")
                
            if idx%12==9:
                TSC200C=TSC200C+float(mean)
                TSC200Cstd=TSC200Cstd+(float(std))**2
                # plt.plot(3, float(mean), marker='.', color=colors[1], markersize=dotsize,alpha=0.75,label="TSC TR 200 ms")
                
            if idx%12==10:
                TSC300U=TSC300U+float(mean)
                TSC300Ustd=TSC300Ustd+(float(std))**2
                # plt.plot(3.5, float(mean), marker='.', color=colors[2], markersize=dotsize,alpha=0.75,label="TSC TR 300 ms")
                
            if idx%12==11:
                TSC300C=TSC300C+float(mean)
                TSC300Cstd=TSC300Cstd+(float(std))**2    
                # plt.plot(3.75, float(mean), marker='.', color=colors[1], markersize=dotsize,alpha=0.75,label="TSC TR 300 ms")
        
        nbmeas=100;
        meanT1=T1m/float(nbmeas)
        stdT1=numpy.sqrt(T1std/float(nbmeas))
        meanVFA=VFA/float(nbmeas)
        stdVFA=numpy.sqrt(VFAstd/float(nbmeas))
        meanTSC100U=TSC100U/float(nbmeas)
        stdTSC100U=numpy.sqrt(TSC100Ustd/float(nbmeas))
        meanTSC100C=TSC100C/float(nbmeas)
        stdTSC100C=numpy.sqrt(TSC100Cstd/float(nbmeas))
        meanTSC120U=TSC120U/float(nbmeas)
        stdTSC120U=numpy.sqrt(TSC120Ustd/float(nbmeas))
        meanTSC120C=TSC120C/float(nbmeas)
        stdTSC120C=numpy.sqrt(TSC120Cstd/float(nbmeas))
        meanTSC150U=TSC150U/float(nbmeas)
        stdTSC150U=numpy.sqrt(TSC150Ustd/float(nbmeas))
        meanTSC150C=TSC150C/float(nbmeas)
        stdTSC150C=numpy.sqrt(TSC150Cstd/float(nbmeas))
        meanTSC200U=TSC200U/float(nbmeas)
        stdTSC200U=numpy.sqrt(TSC200Ustd/float(nbmeas))
        meanTSC200C=TSC200C/float(nbmeas)
        stdTSC200C=numpy.sqrt(TSC200Cstd/float(nbmeas))
        meanTSC300U=TSC300U/float(nbmeas)
        stdTSC300U=numpy.sqrt(TSC300Ustd/float(nbmeas))
        meanTSC300C=TSC300C/float(nbmeas)
        stdTSC300C=numpy.sqrt(TSC300Cstd/float(nbmeas))
        
        print(meanT1,stdT1)
        print(meanVFA,stdVFA)
        print(meanTSC100U,stdTSC100U)
        print(meanTSC100C,stdTSC100C)
        print(meanTSC120U,stdTSC120U)
        print(meanTSC120C,stdTSC120C)
        print(meanTSC150U,stdTSC150U)
        print(meanTSC150C,stdTSC150C)
        print(meanTSC200U,stdTSC200U)
        print(meanTSC200C,stdTSC200C)
        print(meanTSC300U,stdTSC300U)
        print(meanTSC300C,stdTSC300C)
        
        print("-------------------------------")
        print("Processing Images to add tubes data")
        
        import nibabel as nib
    
        MaskTube50_2 = nib.load("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/Tube50-2.nii")
        MaskTube50_2 =MaskTube50_2.get_data()
        MaskTube100_2 = nib.load("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/Tube100-2.nii")
        MaskTube100_2 =MaskTube100_2.get_data()
        MaskTube50_6 = nib.load("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/Tube50-6.nii")
        MaskTube50_6 =MaskTube50_6.get_data()
        MaskTube100_6 = nib.load("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/Tube100-6.nii")
        MaskTube100_6 =MaskTube100_6.get_data()
        
        TSC100U = nib.load("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC100-6_NoT1_Concentration.nii")
        TSC100U =TSC100U.get_data()
        TSC120U = nib.load("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC120-6_NoT1_Concentration.nii")
        TSC120U =TSC120U.get_data()
        TSC150U = nib.load("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC150-6_NoT1_Concentration.nii")
        TSC150U =TSC150U.get_data()
        TSC200U = nib.load("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC200-6_NoT1_Concentration.nii")
        TSC200U =TSC200U.get_data()
        TSC300U = nib.load("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC300-6_NoT1_Concentration.nii")
        TSC300U =TSC300U.get_data()
        
        TSC100C = nib.load("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC100-6_T1VFA_Concentration.nii")
        TSC100C =TSC100C.get_data()
        TSC120C = nib.load("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC120-6_T1VFA_Concentration.nii")
        TSC120C =TSC120C.get_data()
        TSC150C = nib.load("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC150-6_T1VFA_Concentration.nii")
        TSC150C =TSC150C.get_data()
        TSC200C = nib.load("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC200-6_T1VFA_Concentration.nii")
        TSC200C =TSC200C.get_data()
        TSC300C = nib.load("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/TSC300-6_T1VFA_Concentration.nii")
        TSC300C =TSC300C.get_data()
        
        VFAIm = nib.load("C:/Users/AC243636/Pictures/resultats/Sodium/2016/Saltit_project/NewPhantomTestsMarch17/VFA-Tubes6_Concentration.nii")
        VFAIm =VFAIm.get_data()
        
        TSC100U_T50_2 = numpy.zeros(shape=TSC100U.shape)
        TSC100U_T50_6 = numpy.zeros(shape=TSC100U.shape)
        TSC100U_T100_2 = numpy.zeros(shape=TSC100U.shape)
        TSC100U_T100_6 = numpy.zeros(shape=TSC100U.shape)
        
        TSC120U_T50_2 = numpy.zeros(shape=TSC100U.shape)
        TSC120U_T50_6 = numpy.zeros(shape=TSC100U.shape)
        TSC120U_T100_2 = numpy.zeros(shape=TSC100U.shape)
        TSC120U_T100_6 = numpy.zeros(shape=TSC100U.shape)
        
        TSC150U_T50_2 = numpy.zeros(shape=TSC100U.shape)
        TSC150U_T50_6 = numpy.zeros(shape=TSC100U.shape)
        TSC150U_T100_2 = numpy.zeros(shape=TSC100U.shape)
        TSC150U_T100_6 = numpy.zeros(shape=TSC100U.shape)
        
        TSC200U_T50_2 = numpy.zeros(shape=TSC100U.shape)
        TSC200U_T50_6 = numpy.zeros(shape=TSC100U.shape)
        TSC200U_T100_2 = numpy.zeros(shape=TSC100U.shape)
        TSC200U_T100_6 = numpy.zeros(shape=TSC100U.shape)
        
        TSC300U_T50_2 = numpy.zeros(shape=TSC100U.shape)
        TSC300U_T50_6 = numpy.zeros(shape=TSC100U.shape)
        TSC300U_T100_2 = numpy.zeros(shape=TSC100U.shape)
        TSC300U_T100_6 = numpy.zeros(shape=TSC100U.shape)
        
        TSC100C_T50_2 = numpy.zeros(shape=TSC100U.shape)
        TSC100C_T50_6 = numpy.zeros(shape=TSC100U.shape)
        TSC100C_T100_2 = numpy.zeros(shape=TSC100U.shape)
        TSC100C_T100_6 = numpy.zeros(shape=TSC100U.shape)
        
        TSC120C_T50_2 = numpy.zeros(shape=TSC100U.shape)
        TSC120C_T50_6 = numpy.zeros(shape=TSC100U.shape)
        TSC120C_T100_2 = numpy.zeros(shape=TSC100U.shape)
        TSC120C_T100_6 = numpy.zeros(shape=TSC100U.shape)
        
        TSC150C_T50_2 = numpy.zeros(shape=TSC100U.shape)
        TSC150C_T50_6 = numpy.zeros(shape=TSC100U.shape)
        TSC150C_T100_2 = numpy.zeros(shape=TSC100U.shape)
        TSC150C_T100_6 = numpy.zeros(shape=TSC100U.shape)
        
        TSC200C_T50_2 = numpy.zeros(shape=TSC100U.shape)
        TSC200C_T50_6 = numpy.zeros(shape=TSC100U.shape)
        TSC200C_T100_2 = numpy.zeros(shape=TSC100U.shape)
        TSC200C_T100_6 = numpy.zeros(shape=TSC100U.shape)
        
        TSC300C_T50_2 = numpy.zeros(shape=TSC100U.shape)
        TSC300C_T50_6 = numpy.zeros(shape=TSC100U.shape)
        TSC300C_T100_2 = numpy.zeros(shape=TSC100U.shape)
        TSC300C_T100_6 = numpy.zeros(shape=TSC100U.shape)
        
        VFA_T50_2 = numpy.zeros(shape=TSC100U.shape)
        VFA_T50_6 = numpy.zeros(shape=TSC100U.shape)
        VFA_T100_2 = numpy.zeros(shape=TSC100U.shape)
        VFA_T100_6 = numpy.zeros(shape=TSC100U.shape)

        for i in range(TSC100U.shape[0]):
            for j in range(TSC100U.shape[1]):
                for k in range(TSC100U.shape[2]):
                    if(MaskTube50_2[i,j,k]>0):
                        TSC100U_T50_2[i,j,k]=TSC100U[i,j,k]
                        TSC120U_T50_2[i,j,k]=TSC120U[i,j,k]
                        TSC150U_T50_2[i,j,k]=TSC150U[i,j,k]
                        TSC200U_T50_2[i,j,k]=TSC200U[i,j,k]
                        TSC300U_T50_2[i,j,k]=TSC300U[i,j,k]
                        
                        TSC100C_T50_2[i,j,k]=TSC100C[i,j,k]
                        TSC120C_T50_2[i,j,k]=TSC120C[i,j,k]
                        TSC150C_T50_2[i,j,k]=TSC150C[i,j,k]
                        TSC200C_T50_2[i,j,k]=TSC200C[i,j,k]
                        TSC300C_T50_2[i,j,k]=TSC300C[i,j,k]
                        
                        VFA_T50_2[i,j,k]=VFAIm[i,j,k]

                    if(MaskTube100_2[i,j,k]>0):
                        TSC100U_T100_2[i,j,k]=TSC100U[i,j,k]
                        TSC120U_T100_2[i,j,k]=TSC120U[i,j,k]
                        TSC150U_T100_2[i,j,k]=TSC150U[i,j,k]
                        TSC200U_T100_2[i,j,k]=TSC200U[i,j,k]
                        TSC300U_T100_2[i,j,k]=TSC300U[i,j,k]
                        
                        TSC100C_T100_2[i,j,k]=TSC100C[i,j,k]
                        TSC120C_T100_2[i,j,k]=TSC120C[i,j,k]
                        TSC150C_T100_2[i,j,k]=TSC150C[i,j,k]
                        TSC200C_T100_2[i,j,k]=TSC200C[i,j,k]
                        TSC300C_T100_2[i,j,k]=TSC300C[i,j,k]
                        
                        VFA_T100_2[i,j,k]=VFAIm[i,j,k]
        
                    if(MaskTube50_6[i,j,k]>0):
                        TSC100U_T50_6[i,j,k]=TSC100U[i,j,k]
                        TSC120U_T50_6[i,j,k]=TSC120U[i,j,k]
                        TSC150U_T50_6[i,j,k]=TSC150U[i,j,k]
                        TSC200U_T50_6[i,j,k]=TSC200U[i,j,k]
                        TSC300U_T50_6[i,j,k]=TSC300U[i,j,k]
                        
                        TSC100C_T50_6[i,j,k]=TSC100C[i,j,k]
                        TSC120C_T50_6[i,j,k]=TSC120C[i,j,k]
                        TSC150C_T50_6[i,j,k]=TSC150C[i,j,k]
                        TSC200C_T50_6[i,j,k]=TSC200C[i,j,k]
                        TSC300C_T50_6[i,j,k]=TSC300C[i,j,k]
                        
                        VFA_T50_6[i,j,k]=VFAIm[i,j,k]
                        
                    if(MaskTube100_6[i,j,k]>0):
                        TSC100U_T100_6[i,j,k]=TSC100U[i,j,k]
                        TSC120U_T100_6[i,j,k]=TSC120U[i,j,k]
                        TSC150U_T100_6[i,j,k]=TSC150U[i,j,k]
                        TSC200U_T100_6[i,j,k]=TSC200U[i,j,k]
                        TSC300U_T100_6[i,j,k]=TSC300U[i,j,k]
                        
                        TSC100C_T100_6[i,j,k]=TSC100C[i,j,k]
                        TSC120C_T100_6[i,j,k]=TSC120C[i,j,k]
                        TSC150C_T100_6[i,j,k]=TSC150C[i,j,k]
                        TSC200C_T100_6[i,j,k]=TSC200C[i,j,k]
                        TSC300C_T100_6[i,j,k]=TSC300C[i,j,k]

                        VFA_T100_6[i,j,k]=VFAIm[i,j,k]
        
        from processingFunctions import MeanStdCalc3
        maxbound=200
        TSC100U_50_2_val,TSC100U_50_2_mean,TSC100U_50_2_std=MeanStdCalc3(TSC100U_T50_2,0,maxbound)
        TSC120U_50_2_val,TSC120U_50_2_mean,TSC120U_50_2_std=MeanStdCalc3(TSC120U_T50_2,0,maxbound)
        TSC150U_50_2_val,TSC150U_50_2_mean,TSC150U_50_2_std=MeanStdCalc3(TSC150U_T50_2,0,maxbound)
        TSC200U_50_2_val,TSC200U_50_2_mean,TSC200U_50_2_std=MeanStdCalc3(TSC200U_T50_2,0,maxbound)
        TSC300U_50_2_val,TSC300U_50_2_mean,TSC300U_50_2_std=MeanStdCalc3(TSC300U_T50_2,0,maxbound)
        
        TSC100C_50_2_val,TSC100C_50_2_mean,TSC100C_50_2_std=MeanStdCalc3(TSC100C_T50_2,0,maxbound)
        TSC120C_50_2_val,TSC120C_50_2_mean,TSC120C_50_2_std=MeanStdCalc3(TSC120C_T50_2,0,maxbound)
        TSC150C_50_2_val,TSC150C_50_2_mean,TSC150C_50_2_std=MeanStdCalc3(TSC150C_T50_2,0,maxbound)
        TSC200C_50_2_val,TSC200C_50_2_mean,TSC200C_50_2_std=MeanStdCalc3(TSC200C_T50_2,0,maxbound)
        TSC300C_50_2_val,TSC300C_50_2_mean,TSC300C_50_2_std=MeanStdCalc3(TSC300C_T50_2,0,maxbound)
        
        TSC100U_50_6_val,TSC100U_50_6_mean,TSC100U_50_6_std=MeanStdCalc3(TSC100U_T50_6,0,maxbound)
        TSC120U_50_6_val,TSC120U_50_6_mean,TSC120U_50_6_std=MeanStdCalc3(TSC120U_T50_6,0,maxbound)
        TSC150U_50_6_val,TSC150U_50_6_mean,TSC150U_50_6_std=MeanStdCalc3(TSC150U_T50_6,0,maxbound)
        TSC200U_50_6_val,TSC200U_50_6_mean,TSC200U_50_6_std=MeanStdCalc3(TSC200U_T50_6,0,maxbound)
        TSC300U_50_6_val,TSC300U_50_6_mean,TSC300U_50_6_std=MeanStdCalc3(TSC300U_T50_6,0,maxbound)
        
        TSC100C_50_6_val,TSC100C_50_6_mean,TSC100C_50_6_std=MeanStdCalc3(TSC100C_T50_6,0,maxbound)
        TSC120C_50_6_val,TSC120C_50_6_mean,TSC120C_50_6_std=MeanStdCalc3(TSC120C_T50_6,0,maxbound)
        TSC150C_50_6_val,TSC150C_50_6_mean,TSC150C_50_6_std=MeanStdCalc3(TSC150C_T50_6,0,maxbound)
        TSC200C_50_6_val,TSC200C_50_6_mean,TSC200C_50_6_std=MeanStdCalc3(TSC200C_T50_6,0,maxbound)
        TSC300C_50_6_val,TSC300C_50_6_mean,TSC300C_50_6_std=MeanStdCalc3(TSC300C_T50_6,0,maxbound)
        
        
        TSC100U_100_2_val,TSC100U_100_2_mean,TSC100U_100_2_std=MeanStdCalc3(TSC100U_T100_2,0,maxbound)
        TSC120U_100_2_val,TSC120U_100_2_mean,TSC120U_100_2_std=MeanStdCalc3(TSC120U_T100_2,0,maxbound)
        TSC150U_100_2_val,TSC150U_100_2_mean,TSC150U_100_2_std=MeanStdCalc3(TSC150U_T100_2,0,maxbound)
        TSC200U_100_2_val,TSC200U_100_2_mean,TSC200U_100_2_std=MeanStdCalc3(TSC200U_T100_2,0,maxbound)
        TSC300U_100_2_val,TSC300U_100_2_mean,TSC300U_100_2_std=MeanStdCalc3(TSC300U_T100_2,0,maxbound)
        
        TSC100C_100_2_val,TSC100C_100_2_mean,TSC100C_100_2_std=MeanStdCalc3(TSC100C_T100_2,0,maxbound)
        TSC120C_100_2_val,TSC120C_100_2_mean,TSC120C_100_2_std=MeanStdCalc3(TSC120C_T100_2,0,maxbound)
        TSC150C_100_2_val,TSC150C_100_2_mean,TSC150C_100_2_std=MeanStdCalc3(TSC150C_T100_2,0,maxbound)
        TSC200C_100_2_val,TSC200C_100_2_mean,TSC200C_100_2_std=MeanStdCalc3(TSC200C_T100_2,0,maxbound)
        TSC300C_100_2_val,TSC300C_100_2_mean,TSC300C_100_2_std=MeanStdCalc3(TSC300C_T100_2,0,maxbound)
        
        TSC100U_100_6_val,TSC100U_100_6_mean,TSC100U_100_6_std=MeanStdCalc3(TSC100U_T100_6,0,maxbound)
        TSC120U_100_6_val,TSC120U_100_6_mean,TSC120U_100_6_std=MeanStdCalc3(TSC120U_T100_6,0,maxbound)
        TSC150U_100_6_val,TSC150U_100_6_mean,TSC150U_100_6_std=MeanStdCalc3(TSC150U_T100_6,0,maxbound)
        TSC200U_100_6_val,TSC200U_100_6_mean,TSC200U_100_6_std=MeanStdCalc3(TSC200U_T100_6,0,maxbound)
        TSC300U_100_6_val,TSC300U_100_6_mean,TSC300U_100_6_std=MeanStdCalc3(TSC300U_T100_6,0,maxbound)
        
        TSC100C_100_6_val,TSC100C_100_6_mean,TSC100C_100_6_std=MeanStdCalc3(TSC100C_T100_6,0,maxbound)
        TSC120C_100_6_val,TSC120C_100_6_mean,TSC120C_100_6_std=MeanStdCalc3(TSC120C_T100_6,0,maxbound)
        TSC150C_100_6_val,TSC150C_100_6_mean,TSC150C_100_6_std=MeanStdCalc3(TSC150C_T100_6,0,maxbound)
        TSC200C_100_6_val,TSC200C_100_6_mean,TSC200C_100_6_std=MeanStdCalc3(TSC200C_T100_6,0,maxbound)
        TSC300C_100_6_val,TSC300C_100_6_mean,TSC300C_100_6_std=MeanStdCalc3(TSC300C_T100_6,0,maxbound)
        
        VFA_50_2_val,VFA_50_2_mean,VFA_50_2_std=MeanStdCalc3(VFA_T50_2,0,maxbound)
        VFA_100_2_val,VFA_100_2_mean,VFA_100_2_std=MeanStdCalc3(VFA_T100_2,0,maxbound)
        VFA_50_6_val,VFA_50_6_mean,VFA_50_6_std=MeanStdCalc3(VFA_T50_6,0,maxbound)
        VFA_100_6_val,VFA_100_6_mean,VFA_100_6_std=MeanStdCalc3(VFA_T100_6,0,maxbound)
        
        ax = fig.add_subplot(111)
        width=0.1
        indU=[1.5,2.5,3.5,4.5,5.5,6.5]
        indC=[7.5,8.5,9.5,10.5,11.5,12.5]
        
        indU_T50=[1.3,2.3,3.3,4.3,5.3,6.3]
        indC_T50=[7.3,8.3,9.3,10.3,11.3,12.3]
        
        indU_T100=[1.7,2.7,3.7,4.7,5.7,6.7]
        indC_T100=[7.7,8.7,9.7,10.7,11.7,12.7]
        
        valsU=[meanTSC100U,meanTSC120U,meanTSC150U,meanTSC200U,meanTSC300U,meanVFA]
        stdevsU=[stdTSC100U,stdTSC120U,stdTSC150U,stdTSC200U,stdTSC300U,stdVFA]
        
        valsC=[meanTSC100C,meanTSC120C,meanTSC150C,meanTSC200C,meanTSC300C,meanVFA]
        stdevsC=[stdTSC100C,stdTSC120C,stdTSC150C,stdTSC200C,stdTSC300C,stdVFA]
        
        # valsU_T50_6=[TSC100U_50_6_mean,TSC120U_50_6_mean,TSC150U_50_6_mean,TSC200U_50_6_mean,TSC300U_50_6_mean,VFA_50_6_mean]
        # stdevsU_T50_6=[TSC100U_50_6_std,TSC120U_50_6_std,TSC150U_50_6_std,TSC200U_50_6_std,TSC300U_50_6_std,VFA_50_6_std]
        
        # valsC_T50_6=[TSC100C_50_6_mean,TSC120C_50_6_mean,TSC150C_50_6_mean,TSC200C_50_6_mean,TSC300C_50_6_mean,VFA_50_6_mean]
        # stdevsC_T50_6=[TSC100C_50_6_std,TSC120C_50_6_std,TSC150C_50_6_std,TSC200C_50_6_std,TSC300C_50_6_std,VFA_50_6_std]
        
        # valsU_T100_6=[TSC100U_100_6_mean,TSC120U_100_6_mean,TSC150U_100_6_mean,TSC200U_100_6_mean,TSC300U_100_6_mean,VFA_100_6_mean]
        # stdevsU_T100_6=[TSC100U_100_6_std,TSC120U_100_6_std,TSC150U_100_6_std,TSC200U_100_6_std,TSC300U_100_6_std,VFA_100_6_std]
        
        # valsC_T100_6=[TSC100C_100_6_mean,TSC120C_100_6_mean,TSC150C_100_6_mean,TSC200C_100_6_mean,TSC300C_100_6_mean,VFA_100_6_mean]
        # stdevsC_T100_6=[TSC100C_100_6_std,TSC120C_100_6_std,TSC150C_100_6_std,TSC200C_100_6_std,TSC300C_100_6_std,VFA_100_6_std]
        
        valsU_T50_2=[TSC100U_50_2_mean,TSC120U_50_2_mean,TSC150U_50_2_mean,TSC200U_50_2_mean,TSC300U_50_2_mean,VFA_50_2_mean]
        stdevsU_T50_2=[TSC100U_50_2_std,TSC120U_50_2_std,TSC150U_50_2_std,TSC200U_50_2_std,TSC300U_50_2_std,VFA_50_2_std]
        
        valsC_T50_2=[TSC100C_50_2_mean,TSC120C_50_2_mean,TSC150C_50_2_mean,TSC200C_50_2_mean,TSC300C_50_2_mean,VFA_50_2_mean]
        stdevsC_T50_2=[TSC100C_50_2_std,TSC120C_50_2_std,TSC150C_50_2_std,TSC200C_50_2_std,TSC300C_50_2_std,VFA_50_2_std]
        
        valsU_T100_2=[TSC100U_100_2_mean,TSC120U_100_2_mean,TSC150U_100_2_mean,TSC200U_100_2_mean,TSC300U_100_2_mean,VFA_100_2_mean]
        stdevsU_T100_2=[TSC100U_100_2_std,TSC120U_100_2_std,TSC150U_100_2_std,TSC200U_100_2_std,TSC300U_100_2_std,VFA_100_2_std]
        
        valsC_T100_2=[TSC100C_100_2_mean,TSC120C_100_2_mean,TSC150C_100_2_mean,TSC200C_100_2_mean,TSC300C_100_2_mean,VFA_100_2_mean]
        stdevsC_T100_2=[TSC100C_100_2_std,TSC120C_100_2_std,TSC150C_100_2_std,TSC200C_100_2_std,TSC300C_100_2_std,VFA_100_2_std]
        
        
        rects1 = ax.bar(indU, valsU, width,
                color='gray',
                yerr=stdevsU,
                error_kw=dict(elinewidth=2,ecolor='black'))
                
        rects2 = ax.bar(indC, valsC, width,
                color='gray',
                yerr=stdevsC,
                error_kw=dict(elinewidth=2,ecolor='black'))
            
        rects3 = ax.bar(indU_T50, valsU_T50_2, width,
                color='gray',
                yerr=stdevsU_T50_2,
                error_kw=dict(elinewidth=2,ecolor='black'))
                
        rects4 = ax.bar(indC_T50, valsC_T50_2, width,
                color='gray',
                yerr=stdevsC_T50_2,
                error_kw=dict(elinewidth=2,ecolor='black'))    

        rects5 = ax.bar(indU_T100, valsU_T100_2, width,
                color='gray',
                yerr=stdevsU_T100_2,
                error_kw=dict(elinewidth=2,ecolor='black'))
                
        rects6 = ax.bar(indC_T100, valsC_T100_2, width,
                color='gray',
                yerr=stdevsC_T100_2,
                error_kw=dict(elinewidth=2,ecolor='black'))                    
                
        ax.legend( (rects1[0], rects2[0],rects3[0], rects4[0],rects5[0], rects6[0]), ('Liquid U', 'Liquid C','50mM 6% U', '50mM 6% C','100mM 6% U', '100mM 6% C') )        
        
        plt.axvline(x=7, linewidth=2, color = 'k')
        plt.axhline(y=50, linewidth=2, color = 'k')
        plt.axhline(y=75, linewidth=2, color = 'k')
        plt.axhline(y=100, linewidth=2, color = 'k')
        
        plt.ylim(0.0, 130)
        plt.xlim(1, 13)
        # plt.title('Influence of T1 on measured Sodium Concentration')
        # plt.ylabel('[Na] in mmol/L')
        
        fig2 = plt.figure(2)
        
        ax2 = fig2.add_subplot(111)
        width=0.1
        indU=[1.5,2.5,3.5,4.5,5.5]
        indC=[7.5,8.5,9.5,10.5,11.5]
        
        indU_T50=[1.3,2.3,3.3,4.3,5.3]
        indC_T50=[7.3,8.3,9.3,10.3,11.3]
        
        indU_T100=[1.7,2.7,3.7,4.7,5.7]
        indC_T100=[7.7,8.7,9.7,10.7,11.7]
        
        # valsU=[(meanTSC100U-meanTSC300U)/meanTSC300U*100,(meanTSC120U-meanTSC300U)/meanTSC300U*100,(meanTSC150U-meanTSC300U)/meanTSC300U*100,(meanTSC200U-meanTSC300U)/meanTSC300U*100,(meanVFA-meanTSC300U)/meanTSC300U*100]
        valsU=[(meanTSC100U-meanTSC300U)/meanTSC300U*100,(meanTSC120U-meanTSC300U)/meanTSC300U*100,(meanTSC150U-meanTSC300U)/meanTSC300U*100,(meanTSC200U-meanTSC300U)/meanTSC300U*100,(meanVFA-meanTSC300U)/meanTSC300U*100]
        # stdevsU=[stdTSC100U,stdTSC120U,stdTSC150U,stdTSC200U,stdTSC300U,stdVFA]
        
        # valsC=[(meanTSC100C-meanTSC300C)/meanTSC300C*100,(meanTSC120C-meanTSC300C)/meanTSC300C*100,(meanTSC150C-meanTSC300C)/meanTSC300C*100,(meanTSC200C-meanTSC300C)/meanTSC300C*100,(meanVFA-meanTSC300C)/meanTSC300C*100]
        valsC=[(meanTSC100C-meanTSC300C)/meanTSC300C*100,(meanTSC120C-meanTSC300C)/meanTSC300C*100,(meanTSC150C-meanTSC300C)/meanTSC300C*100,(meanTSC200C-meanTSC300C)/meanTSC300C*100,(meanVFA-meanTSC300C)/meanTSC300C*100]
        # stdevsC=[stdTSC100C,stdTSC120C,stdTSC150C,stdTSC200C,stdTSC300C,stdVFA]
        
        # valsU_T50_6=[(TSC100U_50_6_mean-TSC300U_50_6_mean)/TSC300U_50_6_mean*100,(TSC120U_50_6_mean-TSC300U_50_6_mean)/TSC300U_50_6_mean*100,(TSC150U_50_6_mean-TSC300U_50_6_mean)/TSC300U_50_6_mean*100,(TSC200U_50_6_mean-TSC300U_50_6_mean)/TSC300U_50_6_mean*100,(VFA_50_6_mean-TSC300U_50_6_mean)/TSC300U_50_6_mean*100]
        valsU_T50_2=[(TSC100U_50_2_mean-TSC300U_50_2_mean)/TSC300U_50_2_mean*100,(TSC120U_50_2_mean-TSC300U_50_2_mean)/TSC300U_50_2_mean*100,(TSC150U_50_2_mean-TSC300U_50_2_mean)/TSC300U_50_2_mean*100,(TSC200U_50_2_mean-TSC300U_50_2_mean)/TSC300U_50_2_mean*100,(VFA_50_2_mean-TSC300U_50_2_mean)/TSC300U_50_2_mean*100]
        # stdevsU_T50_6=[TSC100U_50_6_std,TSC120U_50_6_std,TSC150U_50_6_std,TSC200U_50_6_std,TSC300U_50_6_std,VFA_50_6_std]
        
        # valsC_T50_6=[(TSC100C_50_6_mean-TSC300C_50_6_mean)/TSC300C_50_6_mean*100,(TSC120C_50_6_mean-TSC300C_50_6_mean)/TSC300C_50_6_mean*100,(TSC150C_50_6_mean-TSC300C_50_6_mean)/TSC300C_50_6_mean*100,(TSC200C_50_6_mean-TSC300C_50_6_mean)/TSC300C_50_6_mean*100,(VFA_50_6_mean-TSC300C_50_6_mean)/TSC300C_50_6_mean*100]
        valsC_T50_2=[(TSC100C_50_2_mean-TSC300C_50_2_mean)/TSC300C_50_2_mean*100,(TSC120C_50_2_mean-TSC300C_50_2_mean)/TSC300C_50_2_mean*100,(TSC150C_50_2_mean-TSC300C_50_2_mean)/TSC300C_50_2_mean*100,(TSC200C_50_2_mean-TSC300C_50_2_mean)/TSC300C_50_2_mean*100,(VFA_50_2_mean-TSC300C_50_2_mean)/TSC300C_50_2_mean*100]
        # stdevsC_T50_6=[TSC100C_50_6_std,TSC120C_50_6_std,TSC150C_50_6_std,TSC200C_50_6_std,TSC300C_50_6_std,VFA_50_6_std]
        
        # valsU_T100_6=[(TSC100U_100_6_mean-TSC300U_100_6_mean)/TSC300U_100_6_mean*100,(TSC120U_100_6_mean-TSC300U_100_6_mean)/TSC300U_100_6_mean*100,(TSC150U_100_6_mean-TSC300U_100_6_mean)/TSC300U_100_6_mean*100,(TSC200U_100_6_mean-TSC300U_100_6_mean)/TSC300U_100_6_mean*100,(VFA_100_6_mean-TSC300U_100_6_mean)/TSC300U_100_6_mean*100]
        valsU_T100_2=[(TSC100U_100_2_mean-TSC300U_100_2_mean)/TSC300U_100_2_mean*100,(TSC120U_100_2_mean-TSC300U_100_2_mean)/TSC300U_100_2_mean*100,(TSC150U_100_2_mean-TSC300U_100_2_mean)/TSC300U_100_2_mean*100,(TSC200U_100_2_mean-TSC300U_100_2_mean)/TSC300U_100_2_mean*100,(VFA_100_2_mean-TSC300U_100_2_mean)/TSC300U_100_2_mean*100]
        # stdevsU_T100_6=[TSC100U_100_6_std,TSC120U_100_6_std,TSC150U_100_6_std,TSC200U_100_6_std,TSC300U_100_6_std,VFA_100_6_std]
        
        # valsC_T100_6=[(TSC100C_100_6_mean-TSC300C_100_6_mean)/TSC300C_100_6_mean*100,(TSC120C_100_6_mean-TSC300C_100_6_mean)/TSC300C_100_6_mean*100,(TSC150C_100_6_mean-TSC300C_100_6_mean)/TSC300C_100_6_mean*100,(TSC200C_100_6_mean-TSC300C_100_6_mean)/TSC300C_100_6_mean*100,(VFA_100_6_mean-TSC300C_100_6_mean)/TSC300C_100_6_mean*100]
        valsC_T100_2=[(TSC100C_100_2_mean-TSC300C_100_2_mean)/TSC300C_100_2_mean*100,(TSC120C_100_2_mean-TSC300C_100_2_mean)/TSC300C_100_2_mean*100,(TSC150C_100_2_mean-TSC300C_100_2_mean)/TSC300C_100_2_mean*100,(TSC200C_100_2_mean-TSC300C_100_2_mean)/TSC300C_100_2_mean*100,(VFA_100_2_mean-TSC300C_100_2_mean)/TSC300C_100_2_mean*100]
        # stdevsC_T100_6=[TSC100C_100_6_std,TSC120C_100_6_std,TSC150C_100_6_std,TSC200C_100_6_std,TSC300C_100_6_std,VFA_100_6_std]
        
        rects1 = ax2.bar(indU, valsU, width, color='gray')
                
        rects2 = ax2.bar(indC, valsC, width, color='gray')
            
        rects3 = ax2.bar(indU_T50, valsU_T50_2, width, color='gray')
                
        rects4 = ax2.bar(indC_T50, valsC_T50_2, width, color='gray')    

        rects5 = ax2.bar(indU_T100, valsU_T100_2, width, color='gray')
                
        rects6 = ax2.bar(indC_T100, valsC_T100_2, width, color='gray')                    
                
        ax2.legend( (rects1[0], rects2[0],rects3[0], rects4[0],rects5[0], rects6[0]), ('Liquid U', 'Liquid C','50mM 6% U', '50mM 6% C','100mM 6% U', '100mM 6% C'), loc=4 )        
        plt.xlim(1, 13)
        
        plt.show()
        
        
def ProcessMeasureCSV_invivo(file):
    import csv, numpy
    with open(file, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        TSC639=0;VFA639=0;T1_639=0;TSC639std=0;VFA639std=0;T1_639std=0;
        TSC643=0;VFA643=0;T1_643=0;TSC643std=0;VFA643std=0;T1_643std=0;
        TSC645=0;VFA645=0;T1_645=0;TSC645std=0;VFA645std=0;T1_645std=0;
        TSC663=0;VFA663=0;T1_663=0;TSC663std=0;VFA663std=0;T1_663std=0;
        for idx, row in enumerate(spamreader):
            # print idx
            a=row[0].split("\t")
            mean=a[3]
            std=a[4]
        
            if idx%12==0:
                TSC639=TSC639+float(mean)
                TSC639std=TSC639std+(float(std))**2
                
            if idx%12==1:
                VFA639=VFA639+float(mean)
                VFA639std=VFA639std+(float(std))**2

            if idx%12==2:
                T1_639=T1_639+float(mean)
                T1_639std=T1_639std+(float(std))**2
            if idx%12==3:
                TSC643=TSC643+float(mean)
                TSC643std=TSC643std+(float(std))**2
            if idx%12==4:
                VFA643=VFA643+float(mean)
                VFA643std=VFA643std+(float(std))**2
            if idx%12==5:
                T1_643=T1_643+float(mean)
                T1_643std=T1_643std+(float(std))**2
            if idx%12==6:
                TSC645=TSC645+float(mean)
                TSC645std=TSC645std+(float(std))**2
            if idx%12==7:
                VFA645=VFA645+float(mean)
                VFA645std=VFA645std+(float(std))**2
            if idx%12==8:
                T1_645=T1_645+float(mean)
                T1_645std=T1_645std+(float(std))**2    
            if idx%12==9:
                TSC663=TSC663+float(mean)
                TSC663std=TSC663std+(float(std))**2    
            if idx%12==10:
                VFA663=VFA663+float(mean)
                VFA663std=VFA663std+(float(std))**2
            if idx%12==11:
                T1_663=T1_663+float(mean)
                T1_663std=T1_663std+(float(std))**2            
        meanTSC639=TSC639/float(90)
        stdTSC639=numpy.sqrt(TSC639std/float(90))
        meanVFA639=VFA639/float(90)
        stdVFA639=numpy.sqrt(VFA639std/float(90))
        meanT1_639=T1_639/float(90)
        stdT1_639=numpy.sqrt(T1_639std/float(90))
        
        meanTSC643=TSC643/float(90)
        stdTSC643=numpy.sqrt(TSC643std/float(90))
        meanVFA643=VFA643/float(90)
        stdVFA643=numpy.sqrt(VFA643std/float(90))
        meanT1_643=T1_643/float(90)
        stdT1_643=numpy.sqrt(T1_643std/float(90))
        
        meanTSC645=TSC645/float(90)
        stdTSC645=numpy.sqrt(TSC645std/float(90))
        meanVFA645=VFA645/float(90)
        stdVFA645=numpy.sqrt(VFA645std/float(90))
        meanT1_645=T1_645/float(90)
        stdT1_645=numpy.sqrt(T1_645std/float(90))
        
        meanTSC663=TSC663/float(90)
        stdTSC663=numpy.sqrt(TSC663std/float(90))
        meanVFA663=VFA663/float(90)
        stdVFA663=numpy.sqrt(VFA663std/float(90))
        meanT1_663=T1_663/float(90)
        stdT1_663=numpy.sqrt(T1_663std/float(90))
        
        print(meanTSC639,stdTSC639)
        print(meanVFA639,stdVFA639)
        print(meanT1_639,stdT1_639)
        print(meanTSC643,stdTSC643)
        print(meanVFA643,stdVFA643)
        print(meanT1_643,stdT1_643)
        print(meanTSC645,stdTSC645)
        print(meanVFA645,stdVFA645)
        print(meanT1_645,stdT1_645)
        print(meanTSC663,stdTSC663)
        print(meanVFA663,stdVFA663)
        print(meanT1_663,stdT1_663)
        
def SimuSAR():

    B=3;
    tau=500e-6;
    import numpy
    TRmax=200;
    FAmax=90;
    SARmap= numpy.zeros(shape=(TRmax,FAmax))
    
    for TR in range(TRmax):
        for FA in range(FAmax):
            SARmap[TR,FA]= ((B**2)*((FA+1)**2))/(tau*(TR+1)/1000)
    
    from nifty_funclib import SaveArrayAsNIfTI
    SaveArrayAsNIfTI(SARmap,1,1,1,"C:/Users/AC243636/Pictures/SARSimuTest.nii")    
    
    
def ComputeXDensity3DNoB1(FA1,Img_FA1,FA2,Img_FA2):

    import nibabel as nib
    import numpy, os, sys
    from scipy import stats
    from visualization import PlotReconstructedImage
    path=Img_FA1
    Img_FA1 = nib.load(Img_FA1)
    Img_FA1 =Img_FA1.get_data()    
    Img_FA2 = nib.load(Img_FA2)
    Img_FA2 =Img_FA2.get_data()

    TR=0.2                    
    TE=0.0003                

    print(FA1, FA2)

    # PlotReconstructedImage((mask[:,:]))
    # PlotReconstructedImage((Img_FA1[:,:]))
    # PlotReconstructedImage((Img_FA2[:,:]))
    
    T1 = numpy.zeros(shape=Img_FA1.shape)
    M0 = numpy.zeros(shape=Img_FA1.shape)
    
    # for i in range(len(FAMap1[2])):
        # for j in range(len(FAMap1[1])):
            # for k in range(len(FAMap1[0])):
    for i in range(Img_FA1.shape[0]):
        for j in range(Img_FA1.shape[1]):
            for k in range(Img_FA1.shape[2]):
                y = numpy.zeros(2)
                x = numpy.zeros(2)
            
                y[0]=Img_FA1[i,j,k]/numpy.sin(FA1*numpy.pi/180.0)
                y[1]=Img_FA2[i,j,k]/numpy.sin(FA2*numpy.pi/180.0)
                x[0]=Img_FA1[i,j,k]/numpy.tan(FA1*numpy.pi/180.0)
                x[1]=Img_FA2[i,j,k]/numpy.tan(FA2*numpy.pi/180.0)
                # print y
                # slope, intercept, r_value, p_value, std_err = stats.linregress(y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                # print r_value**2, p_value, std_err
                # slope = (y[1]-y[0])/(x[1]-x[0])
                # intercept = y[0]-slope*x[0]
                if not numpy.isnan(-TR/numpy.log(slope)):
                    if -TR/numpy.log(slope) > 14:
                        T1[i,j,k]=14
                    if -TR/numpy.log(slope) < 2:
                        T1[i,j,k]=2
                    else:
                        T1[i,j,k]=-TR/numpy.log(slope)
                else:
                    T1[i,j,k]=0
                M0[i,j,k]=intercept/((1-slope))


    from nifty_funclib import SaveArrayAsNIfTI
    Hpath, Fname = os.path.split(str(path))
    Fname = Fname.split('.')
    OutputPathT1 = os.path.join( Hpath + '\\' + "T1-3D.nii")
    OutputPathM0 = os.path.join( Hpath + '\\' + "M0-3D.nii")
    SaveArrayAsNIfTI(T1,1,1,1,OutputPathT1)    
    SaveArrayAsNIfTI(M0,1,1,1,OutputPathM0)    

def ComputeNoiseM0mapVFA(FA1,Img_FA1,FA2,Img_FA2,RatioMap):

    import nibabel as nib
    import numpy, os, sys
    from scipy import stats
    from visualization import PlotReconstructedImage
    path=Img_FA1
    Img_FA1 = nib.load(Img_FA1)
    Img_FA1 =Img_FA1.get_data()    
    Img_FA2 = nib.load(Img_FA2)
    Img_FA2 =Img_FA2.get_data()    
    RatioMap = nib.load(RatioMap)
    RatioMap =RatioMap.get_data()

    SigmaMap = numpy.zeros(shape=Img_FA1.shape)
    
    for i in range(Img_FA1.shape[0]):
        for j in range(Img_FA1.shape[1]):
            for k in range(Img_FA1.shape[2]):
                A = numpy.sin(FA2*RatioMap[i,j,k])*numpy.tan(FA1*RatioMap[i,j,k]) - numpy.sin(FA1*RatioMap[i,j,k])*numpy.tan(FA2*RatioMap[i,j,k])
                B = numpy.tan(FA1*RatioMap[i,j,k])*numpy.sin(FA1*RatioMap[i,j,k])*(numpy.sin(FA2*RatioMap[i,j,k])-numpy.tan(FA2*RatioMap[i,j,k]))
                C = numpy.tan(FA2*RatioMap[i,j,k])*numpy.sin(FA2*RatioMap[i,j,k])*(numpy.sin(FA1*RatioMap[i,j,k])-numpy.tan(FA1*RatioMap[i,j,k]))
                SigmaMap[i,j,k] = A*(numpy.sqrt(B**2*Img_FA2[i,j,k]**4+C**2*Img_FA1[i,j,k]**4))/((B*Img_FA2[i,j,k]-C*Img_FA1[i,j,k])**2)
                
    from nifty_funclib import SaveArrayAsNIfTI
    Hpath, Fname = os.path.split(str(path))
    Fname = Fname.split('.')
    OutputPathSigma = os.path.join( Hpath + '\\' + "SigmaaTest.nii")
    SaveArrayAsNIfTI(SigmaMap,1,1,1,OutputPathSigma)

def ComputeNoiseM0mapVFAnoB1(FA1,Img_FA1,FA2,Img_FA2):

    import nibabel as nib
    import numpy, os, sys
    from scipy import stats
    from visualization import PlotReconstructedImage
    path=Img_FA1
    Img_FA1 = nib.load(Img_FA1)
    Img_FA1 =Img_FA1.get_data()    
    Img_FA2 = nib.load(Img_FA2)
    Img_FA2 =Img_FA2.get_data()    

    SigmaMap = numpy.zeros(shape=Img_FA1.shape)
    
    for i in range(Img_FA1.shape[0]):
        for j in range(Img_FA1.shape[1]):
            for k in range(Img_FA1.shape[2]):
                A = numpy.sin(FA2)*numpy.tan(FA1) - numpy.sin(FA1)*numpy.tan(FA2)
                B = numpy.tan(FA1)*numpy.sin(FA1)*(numpy.sin(FA2)-numpy.tan(FA2))
                C = numpy.tan(FA2)*numpy.sin(FA2)*(numpy.sin(FA1)-numpy.tan(FA1))
                SigmaMap[i,j,k] = A*(numpy.sqrt(B**2*Img_FA2[i,j,k]**4+C**2*Img_FA1[i,j,k]**4))/((B*Img_FA2[i,j,k]-C*Img_FA1[i,j,k])**2)
                
    from nifty_funclib import SaveArrayAsNIfTI
    Hpath, Fname = os.path.split(str(path))
    Fname = Fname.split('.')
    OutputPathSigma = os.path.join( Hpath + '\\' + "SigmaaTestNoB1.nii")
    SaveArrayAsNIfTI(SigmaMap,1,1,1,OutputPathSigma)                