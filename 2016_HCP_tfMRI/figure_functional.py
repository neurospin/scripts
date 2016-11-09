# -*- coding: utf-8 -*-

import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import os, shutil, glob
import pandas as pd

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
    file_contrast = '/neurospin/brainomics/2016_HCP/functional_analysis/HCP_MMP1.0/contrast_names.csv'
    df_contrast = pd.read_csv(file_contrast)
    df_contrast['index'] = df_contrast['Task']+df_contrast['CopeNumber'].astype('str')
    df_contrast.index = df_contrast['index']
    tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'RELATIONAL', 'SOCIAL', 'WM']
    COPE_NUMS = [[1,3], [1,3], [1,4], [1,4], [1,3], [1, 22]]
    #BONFTHRESHOLD = 5e-2/1 
    for BONF in [180, 360, 1]:
        OUTPUT_SUMMARY = '/neurospin/brainomics/2016_HCP/functional_analysis/FIGURES_SUMMARY_BONF'+str(BONF)
        if not os.path.exists(OUTPUT_SUMMARY):
            os.makedirs(OUTPUT_SUMMARY)
        for j, task in enumerate(tasks):
            for i in range(COPE_NUMS[j][0],COPE_NUMS[j][1]+1):
                select = "mean"
                INPUT = '/neurospin/brainomics/2016_HCP/functional_analysis/FIGURES/pheno_'+select+'_value/'+task+'_'+str(i)+'/'
                os.system('ls '+INPUT+'/snapshot*.png|while read input; do convert  $input -trim /tmp/toto.png; convert  /tmp/toto.png -transparent white $input; done')
                os.system('rm /tmp/toto.png')   
                OUTPUT = '/neurospin/brainomics/2016_HCP/functional_analysis/FIGURES/pheno_'+select+'_value/'+task+'_'+str(i)+'/results_poster/'
                if not os.path.exists(OUTPUT):
                    os.makedirs(OUTPUT)
                file_out = "HCP_"+task+"_"+df_contrast.loc[task+str(i)]['ContrastName']+"_fMRI_Bonf"+str(BONF)+"corrected_poster.png"
                output_image = os.path.join(OUTPUT, file_out)
                images = []
                view = ['extern', 'intern', 'bottom', 'top']
                features = ['herit', 'pval'] 
                sides = ['L', 'R']
                for feature in features:
                    for side in sides:
                        for sd in view:
                            image = INPUT+'snapshot_bonf'+str(BONF)+'_'+feature+'_'+side+'_'+sd+'.png'
                            images.append(image)

                padding = 6
                create_poster( images,  output_image, padding=6, tile=8, align_grid = False)


                ## Adding the colorbar
                image = PIL.Image.open( output_image )
                features = ['herit2', 'pval5']
                colorbar = {}
                max_large = 0
                for feature in features:
                    if feature != None:
                        colorbar[feature] = PIL.Image.open('/home/yl247234/Images/PALETTES/palette_'+feature+'.png')
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
                features = ['H2', 'Log10 Pval']
                #features = ['mu', 'std']
                #features = ['H2 '+pheno, 'Log10 Pval']
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
                path_out = os.path.join(OUTPUT_SUMMARY, file_out)
                shutil.copy(output_image, path_out)
