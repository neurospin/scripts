"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2015
"""
import pandas as pd
import numpy as np
import os, glob, re

## INPUTS ###
#right 'R' or left 'L'
side = 'L'
# directory and filename of the MEGHA.m output file
directory = '/neurospin/brainomics/2016_sulcal_depth/megha/all_features/depthMaxtol0.02/'
filename = 'covar_GenCit5PCA_ICV_MEGHAMEGHAstat.txt'#covar_GenCitHan5PCA_ICV_MEGHAMEGHAstat.txt'
THRESHOLD_PVAL = 1e-1
# feature display 'h2' or 'LogPval'
FEATURE_DISPLAY = 'h2' 
# interval display for h2 [0,1.0] recommended and for LogPval [0,5.0] to adapt
INTERVAL = [0,1.0] 

#### CONSTANTS #####
maf = 0.01
vif = 10.0
sides = {'R': 'right',
         'L': 'left'}
# Load the right hemisphere folds mesh, associated graph and cortical surface 
s_id_antoine = '000001311901'
s_id_y = '000000106871'
city = 'Dresden/'#Mannheim/'#
s_id = s_id_y#antoine
feature = '_depthMax'

# To import the environment variable for anatomist
# which -a anatomist
# . ../bv_env.sh
import anatomist.api 
ana = anatomist.api.Anatomist()
# this module path is added only after Anatomist is initialized
import paletteViewer
window = ana.createWindow('3D')


def updateWindow(obj):
    window.addObjects(obj)
    obj.setChanged()
    obj.notifyObservers()
    """# update view so that its state is OK
    w.refreshNow()"""



### Look for the heritability and pvalues ###
sulcus_names = []
variance_explained = []
p_values  = []
std_variance_explained = []   


df = pd.read_csv(directory+filename, delim_whitespace=True)
df.index = df['Phenotype']
for ind in df.index:
    if sides[side] in ind:
        sulcus = ind[:len(ind)-(1+len(sides[side])+len(feature))]
        #print "Sulcus: " + str(sulcus)
        sulcus_names.append(sulcus)
        variance_explained.append(df.loc[ind]['h2'])
        p_values.append(df.loc[ind]['Pval'])
       

index_p_values = np.nonzero(np.less(p_values, THRESHOLD_PVAL))[0]
variance_explained_selected = []
p_values_selected  = []
sulcus_names_selected = []
for j in range(len(index_p_values)):
    variance_explained_selected.append(variance_explained[index_p_values[j]])
    p_values_selected.append(p_values[index_p_values[j]])
    sulcus_names_selected.append(sulcus_names[index_p_values[j]])
df =  pd.DataFrame({'Sulci': np.asarray(sulcus_names_selected),
                    'h2': np.asarray(variance_explained_selected),
                    'LogPval': -np.log10(np.asarray(p_values_selected))
                })
df.index = df['Sulci']

#### DEFINING THE PATH TO GET THE REPRESENTATION OF THE MAILLAGE #####

subject_folder = "/neurospin/imagen/workspace/cati/BVdatabase/"+city+ s_id+ "/"
file_folds_mesh = os.path.join(subject_folder, "t1mri", "BL",
                  "default_analysis", "folds", "3.1", "default_session_auto",
                  side+ s_id+ "_default_session_auto.data",
                          "aims_Tmtktri.gii")
file_graph = os.path.join(subject_folder, "t1mri", "BL",
                      "default_analysis", "folds", "3.1", "default_session_auto",
                      side+ s_id+ "_default_session_auto.arg")
file_white_mesh = os.path.join(subject_folder, "t1mri", "BL",
                           "default_analysis", "segmentation", "mesh",
                           ""+ s_id+ "_"+side+"white.gii")


sulci = ana.loadObject(file_graph)
sulci.setColorMode(sulci.PropertyMap)


graph = ana.toAimsObject(sulci)
# graph.keys()
# graph.vertices()

# Associate the weights to the label
#weights = {}
for vertex in graph.vertices():
    if vertex.has_key('label'):
        short_label = vertex['label'].replace('.', '')
        short_label = short_label[:len(short_label)-(len(sides[side])+1)]
        if short_label in df.index:
            vertex['heritability'] = df[FEATURE_DISPLAY][short_label]
        else:
            vertex['heritability'] = 0

sulci.setColorProperty('heritability')
sulci.setPalette('actif_ret', minVal=INTERVAL[0], maxVal=INTERVAL[1], absoluteMode=True)
updateWindow(sulci)
white_mesh = ana.loadObject(file_white_mesh)
updateWindow(white_mesh)

# show/hide palette
paletteViewer.ShowHidePaletteCallback().doit(anatomist.api.cpp.set_AObjectPtr(
    [sulci.getInternalRep()]))

ana.execute('WindowConfig', windows=[window], cursor_visibility=0)
ana.execute('WindowConfig', windows=[window], snapshot='/tmp/snapshot.jpg')

# save palette figure
gw = window.parent().findChild(paletteViewer.GroupPaletteWidget)
fig = gw.get(paletteViewer.getObjectId(sulci)).findChild(paletteViewer.PaletteWidget).figure
fig.savefig('/tmp/palette.png')

