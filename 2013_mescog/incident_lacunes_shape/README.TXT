is223373 64G 16CPU

M000 : Volume
Mxyz :Moment non centrés
M100 somme des x

(M100, M010, M001) / M000 central Mass

uxyz :Moment centrés réduit, invriant en échelle et position, sensible en rotation
u100 : Sigma(x - xc) / M000
u200 : Sigma(x - xc)^2 / M000
u110 : Sigma(x - xc)(y - yc) / M000 ^ (puissance)

In [0..11]: Invariant de moments 3D

x+y+z=1: ordre 1
x+y+z=2 ordre 2

Ouput of AimsMoment -i 096_2100-M36_LacBL.nii.gz
================================================

reading data 096_2100-M36_LacBL.nii.gz...
done
Results for label :  
Image geometry: (0.8984x1.2002x0.898407)
Volume :  24.2178mm3
Number of points : 25
Volume (mm3) : 24.2178 :M000

Order 1 moments : 2167.02 3024.04 2285.4 
M100, M010, M001

Order 2 moments : 18.7649 26.4568 11.6969 -11.0719 -2.03288 1.00275 
u200, u020, u002, u110, u101, u011


Order 3 moments : -2.52875 7.91019 -0.559711 3.19055 -1.54536 -5.70651 2.71187 0.848551 -1.57353 -1.00597 
u300, u030, u003, u210, u201, u120, u021, u102, u012, u111

Center of Mass :  (89.4806,124.868,94.3687)
Orientation :
    Inerty :  34.5066 ; 12.323 ; 10.0889
    V1 :  (-0.580621,0.809477,0.0873327)
    V2 :  (0.445914,0.405912,-0.797745)
    V3 :  (-0.681205,-0.424244,-0.596638)

Moment invariant: 
0.576001 0.285514 -0.273623 -0.267788 -0.222788 -0.250372 0.239844 -0.216948 0.215906 0.26996 -0.247557 0.225748 

Invariant de moments (Invariant en transaltion, rotation echelle)
I[0..11] p. 101 thèse Fabrice


##################################################################
## QC
More than one label /home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_images/023/023_1037-M36_LacBL.nii.gz

################################################################################
DISCUSSION
==========

Moments
-------

- incident_lacunes_moments.csv: all computed moments
- incident_lacunes_moments_descriptive.xls: some descriptive stats on the moment
with signification

Methods
-------

PCA

PCA on 11 Invariant's moments (pure shape descriptors)
- pc1-pc2_annotated.svg
- pc1-pc2_clusters.csv (columns PC1, PC2)

Clustering
~~~~~~~~~~

Plot of lacunes using the two first components (70% of explained variance).
The plot highlight four patterns defined by 4 lines. This lines where
used to define 4 groups.
Clustering using the 
- pc1-pc2_clusters.csv (columns label)
- pc1-pc2_clusters.svg
- pc1-pc2_clusters_annotated.svg

Results
-------
The plot within the two first PCs highlight one axis (diagonal top-left -> bottom-right, y = -x)
that explain an anisotropy of the shape.
- pc1-pc2_color-by-inerty_max_norm_from-mesh.svg shows that shape anisotropy increased
moving bottom-right to top-left.
The anisotropy is measured using the max inerty^2 / sum of inerties^2 ratio
The shape anisotropy explain 24% of the total shape variance (F=169.38, p<10e-16, Manova Pillai test)
However, the orthogonal axis is unclear

)measure fdsfds
fds
 

