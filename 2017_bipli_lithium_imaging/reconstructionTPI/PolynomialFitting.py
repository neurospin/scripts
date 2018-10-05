# Polynomial Fitting (Field Interpolation) from data 
# Author : Arthur Coste, from A. Vignaud's matlab code transposed into Python

def polynomialFitting(Image, mask, order,save):

    # Image, typically B1Maps : 3D image to be fitted with polynomial (can be complex, but only its magnitude is considered)
    # mask : 3D image of the region to be fitted (typically the brain)
    # order: order of the polynomial (between 6 and 8 is good for B1 maps)
    # Nch : number of channels
    # default parameters, if no mask provided == Over the whole image and default Interpolation order = 4
    
    import numpy, os, sys
    import nibabel as nib
    Image = nib.load(Image)
    Image =Image.get_data()    
    Image =numpy.squeeze(Image)
    # Image = Image[:,:,7]
    # Dims=Image.shape
    if mask == [] :
        mask = numpy.ones(shape=(Image.shape))
    else:
        mask = nib.load(mask)
        mask = mask.get_data()    
        mask = numpy.squeeze(mask)
    if order == [] :
        order = 4
        
    if save == [] :
        save = False    
        
    if len(Image.shape) ==2 :
        print("2D Mode")
        resx = 2./(Image.shape[0]+1); resy = 2./(Image.shape[1]+1); resz = 2;
        X=numpy.zeros(shape=Image.shape);Y=numpy.zeros(shape=Image.shape);Z = numpy.zeros(shape=(Image.shape))
        X,Y = numpy.meshgrid(numpy.arange(-1.0+ resx/2,1.0- resx/2,resx,dtype=float),numpy.arange(-1.0+ resy/2,1.0- resy/2,resy,dtype=float))
    if len(Image.shape) ==3 :
        print("3D Mode")
        resx = 2./Image.shape[0]; resy = 2./Image.shape[1]; resz = 2./Image.shape[2];
        X=numpy.zeros(shape=Image.shape);Y=numpy.zeros(shape=Image.shape);Z = numpy.zeros(shape=(Image.shape))
        X,Y,Z = numpy.meshgrid(numpy.arange(-1.0+ resx/2,1.0- resx/2,resx,dtype=float),numpy.arange(-1.0+ resy/2,1.0- resy/2,resy,dtype=float),numpy.arange(-1.0+ resz/2,1.0- resz/2,resz,dtype=float))
    Index = numpy.isnan(mask)
    mask[Index]=0
    I = numpy.where(mask[:])
    Xvec=numpy.ravel(X); Yvec=numpy.ravel(Y); Zvec=numpy.ravel(Z);
    A=(numpy.power(Xvec,(1)))*(numpy.power(Yvec,(1)))*(numpy.power(Zvec,(1)))

    P=numpy.zeros(shape=(Xvec.size,order**3))

    for k in range(order):
        for m in range(order):
            for n in range(order):
                P[:,(k)*(order)**2 + (m)*(order) + n] = (Xvec**k)*(Yvec**m)*(Zvec**n);

    Q= numpy.linalg.pinv(P, rcond=1e-12)
    Bvec=numpy.absolute(Image);
    D=Bvec[I]
    D=numpy.expand_dims(D,axis=1)
    print(Q.shape)
    print(D.shape)
    c = numpy.dot(Q,D);
    Bvecout=numpy.dot(P,c);

    if Image.ndim ==2 :
        InterpolatedImage=numpy.reshape(Bvecout,(Image.shape[0],Image.shape[1]))
        from visualization import PlotReconstructedImage
        PlotReconstructedImage(Image)    
        PlotReconstructedImage(InterpolatedImage)
        if save:
            from nifty_funclib import SaveArrayAsNIfTI
            Hpath, Fname = os.path.split(str(OutputPath))
            Fname = Fname[0].split('.')
            OutputPath = os.path.join( Hpath + '\\' + Fname[0] + 'Interpolated_Image_order_'+order+'.nii')
            SaveArrayAsNIfTI(InterpolatedImage,1,1,1,OutputPath)
        
    if Image.ndim ==3 :
        InterpolatedImage=numpy.reshape(Bvecout,(Image.shape[0],Image.shape[1],Image.shape[2]))
        PlotReconstructedImage(InterpolatedImage)
        if save:
            from nifty_funclib import SaveArrayAsNIfTI
            Hpath, Fname = os.path.split(str(OutputPath))
            Fname = Fname[0].split('.')
            OutputPath = os.path.join( Hpath + '\\' + Fname[0] + 'Interpolated_Image_order_'+order+'.nii')
            SaveArrayAsNIfTI(InterpolatedImage,1,1,1,OutputPath)
    return InterpolatedImage

