# -*- coding: utf-8 -*-

import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import glob
import os

def create_poster(images, output_image, tile=4, padding=4, bgcolor=(255,255,255),
                  align_grid=True):
    # print 'calculating poster size...'
    rowsizes = []
    maxsize = [0, 0]
    n = 0
    for imagename in images:
        image = PIL.Image.open( imagename )
        if n % tile == 0:
            rowsizes.append( ( 0, 0 ) )
        rowsizes[-1] = ( rowsizes[-1][0] + image.size[0] + padding,
            max( rowsizes[-1][1], image.size[1] + padding ) )
        maxsize = (max((maxsize[0], image.size[0])),
                   max((maxsize[1], image.size[1])))
        n += 1
    print 'sizes:', rowsizes
    if align_grid:
        size = ((maxsize[0] + padding) * min(tile, len(images)) - padding,
                (maxsize[1] + padding) * len(rowsizes) - padding)
    else:
        size = ( max( [ rs[0] for rs in rowsizes ] ) - padding,
            sum( [ rs[1] for rs in rowsizes ] ) - padding )

    print 'size:', size
    outimage = PIL.Image.new( 'RGB', size, bgcolor )
    print 'creating image...'
    n = 0
    line = 0
    xpos = 0
    ypos = 0
    #rowsizes.insert( 0, ( 0, 0 ) )
    for imagename in images:
        image = PIL.Image.open( imagename )
        if align_grid:
            bbox = ((maxsize[0] + padding) * n, (maxsize[1] + padding) * line,
                    0, 0)
            bbox = (bbox[0], bbox[1],
                    bbox[0] + image.size[0], bbox[1] + image.size[1])
        else:
            y = ypos + ( rowsizes[line][1] - image.size[1] ) / 2
            bbox = ( xpos, y, xpos+image.size[0], y + image.size[1] )
        outimage.paste( image, bbox )
        xpos = bbox[2] + padding
        n += 1
        if n == tile:
            n = 0
            line += 1
            xpos = 0
            ypos = bbox[3] + padding
    path = os.path.dirname(output_image)
    outimage.save( open( output_image, 'w' ) )
    # print 'done.'


if __name__ == '__main__':
    database_parcel = 'hcp'
    base = 'Freesurfer_new' # Freesurfer or BV
    pheno = 'DPF'
    INPUT = '/home/yl247234/Images/final_snap_sym/group_'+database_parcel+'_'+base+'/'
    os.system('ls '+INPUT+'/snapshot*.jpg|while read input; do convert  $input -trim /tmp/toto.jpg; convert  /tmp/toto.jpg -transparent white $input; done')
    os.system('rm /tmp/toto.jpg')
    OUTPUT = '/home/yl247234/Images/final_snap_sym/results_update/asym_'+pheno+'_'
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
    output_image = OUTPUT+database_parcel+'_'+base+'_poster.jpg'
    images = []

    features = ['density_smooth', 'clusters', 'frequency']
    sides = ['L', 'R']
    view = ['extern', 'intern']
    for feature in features:
        for side in sides:
            for sd in view:
                image = INPUT+'snapshot_'+feature+'_'+side+'_'+sd+'.jpg'
                images.append(image)
    if "sym" in INPUT:
        feature = 'frequency_asym'
        for side in sides:
            for sd in view:
                image = INPUT+'snapshot_'+feature+'_'+sd+'.jpg'
                images.append(image)
    features = [pheno, 'pval_'+pheno] 
    sides = ['Left', 'Right']
    for feature in features:
        for side in sides:
            for sd in view:
                image = INPUT+'snapshot_'+feature+'_'+side+'_'+sd+'.jpg'
                images.append(image)
    create_poster( images,  output_image, tile=4, align_grid = True)
    

    ## Adding the colorbar
    image = PIL.Image.open( output_image )
    if "sym" in INPUT:
        features = ['density', None, 'frequency', 'asym', 'herit', 'pval']
    else:
        features = ['density', None, 'frequency', 'herit', 'pval']
    padding = 4
    colorbar = {}
    max_large = 0
    for feature in features:
        if feature != None:
            colorbar[feature] = PIL.Image.open('/home/yl247234/Images/new_snap/palette_'+feature+'.png')
            if int(colorbar[feature].size[0]) > max_large:
                max_large = int(colorbar[feature].size[0])
            colorbar[feature] = colorbar[feature].resize((int(colorbar[feature].size[0]), int((image.size[1]-(len(features)-1)*padding)/len(features))))
    bgcolor = (255,255,255)
    size =( image.size[0]+padding+max_large, 
            image.size[1])
    outimage = PIL.Image.new( 'RGB', size, bgcolor)
    bbox = (0, 0, image.size[0], image.size[1])
    outimage.paste( image, bbox )
    for k,feature in enumerate(features):
        if feature != None:
            bbox = (image.size[0]+padding, int(image.size[1]*k/len(features))+k*padding, size[0], int(image.size[1]*k/len(features))+k*padding+int((image.size[1]-(len(features)-1)*padding)/len(features)))
            outimage.paste( colorbar[feature], bbox )

    path = os.path.dirname(output_image)
    outimage.save( open( output_image, 'w' ) )  

    # Writing the name of the features
    draw = PIL.ImageDraw.Draw(outimage)
    path_font ='/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'
    font = PIL.ImageFont.truetype(path_font, 50)
    if "sym" in INPUT:
        features = ['Density smoothed', 'Clusters', 'Frequency', 'Freq asymmetry', 'H2 '+pheno, 'Log10 Pval']
    else:
        features = ['Density smoothed', 'Clusters', 'Frequency', 'H2 '+pheno, 'Log10 Pval']
    max_length = 0
    for feature in features:
        feature = feature.replace('_', ' ')
        max_length = max( max_length, draw.textsize(feature , font=font)[0])
    outimage2 = PIL.Image.new( 'RGB', (size[0]+max_length+4*padding,size[1]), bgcolor)
    draw = PIL.ImageDraw.Draw(outimage2)
    bbox = (max_length+4*padding,0,size[0]+max_length+4*padding, size[1])
    outimage2.paste( outimage, bbox )
    maxsize_image = (size[1]-padding*(len(features)-1))/len(features)
    for k,feature in enumerate(features):
        feature = feature.replace('_', ' ')
        size_text = draw.textsize(feature , font=font)
        position = (2*padding+(max_length-size_text[0])/2, k*(maxsize_image+padding)+(maxsize_image-size_text[1])/2)
        draw.text(position , feature, fill=(0, 0, 0), font=font)
    outimage2.save( open( output_image, 'w' ) )  
