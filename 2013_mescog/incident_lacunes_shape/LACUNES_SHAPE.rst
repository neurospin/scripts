======================
Incident Lacunes Shape
======================

:Author: Edouard.Duchesnay@cea.fr
:Date: 2014/09/15

Characterization of incident lacunes shape using Invariant's moments and tensor's invariant.

.. contents::

Produced output (Material)
==========================

In root directory
-----------------

- ``incident_lacunes_moments.csv``: all computed moments
- ``incident_lacunes_moments_descriptive.xls``: some descriptive stats on the moment

Computed moments:

=====================================   ====================================================
Name                                    Comment
=====================================   ====================================================
lacune_id                               lacune id
number_of_points                        -
vol_mm3                                 -
order_1_monent_0                        non-invariant moment
...                                     non-invariant moment
order_3_monent_9                        non-invariant moment
                                        
center_of_mass_0                        -
center_of_mass_1                        Center of mass coordinates
center_of_mass_2                        -
                                        
orientation_inertia_0                   Max inertia
orientation_inertia_1                   Medium inertia
orientation_inertia_2                   Min inertia
orientation_v1_0                        Max inertia orientation
orientation_v1_1                        Max inertia orientation
orientation_v1_2                        Max inertia orientation
orientation_v2_0                        Medium inertia orientation
orientation_v2_1                        Medium inertia orientation
orientation_v2_2                        Medium inertia orientation
orientation_v3_0                        Min inertia orientation
orientation_v3_1                        Min inertia orientation
orientation_v3_2                        Min inertia orientation
                                        
moment_invariant_0                      moment invariant 0
moment_invariant_1                      -
moment_invariant_2                      -
moment_invariant_3                      -
moment_invariant_4                      -
moment_invariant_5                      Do not consider this moment (Sun et al. MICCAI)
moment_invariant_6                      -
moment_invariant_7                      -
moment_invariant_8                      -
moment_invariant_9                      Do not consider this moment (Sun et al. MICCAI)
moment_invariant_10                     -
moment_invariant_11                     -
                                        
area_mesh                               Lacune area calculated from a mesh
vol_mesh                                Lacune volume calculated from a mesh
perfo_orientation_0                     Perforator orientation
perfo_orientation_1                     Perforator orientation
perfo_orientation_2                     Perforator orientation
                                        
compactness                             vol_mesh^(2/3) / area_mesh

tensor_invariant_fa                     tensor invariant: Fractional Anisotropy
tensor_invariant_mode                   tensor invariant: mode
tensor_invariant_linear_anisotropy      tensor invariant: linear_anisotropy
tensor_invariant_planar_anisotropy      tensor invariant: planar_anisotropy
tensor_invariant_spherical_anisotropy   tensor invariant: spherical_anisotropy
perfo_angle_inertia_max                 angle between perforator and max inertia
=====================================   ====================================================


In ``results_moments_invariant`` directory
------------------------------------------

- ``pca.csv``: PCA components from moments' invariant. It is a csv file 
    of dimension: n_lacunes x [lacune_id, PC1 (first component value), ..., PC10 (last component value)]
- ``pca.txt``: Explained variance ratio of the first two component is 81% + (PCA Loadings, weights vector).
- ``pc1-pc2.pdf``: Many dot plot of lacunes (as dot) ploted in the two first components, 
    annotated with lacune_id and colored with tensor's invariant value.


Other files show lacunes ploted in the two first components colored by FA, 
linear, planar and spherical anisotropy. File suffixed ``with-meshed-lacunes`` plot lacunes instead of simple dots.

Colored by FA:

- ``pc1-pc2_fa.svg``
- ``pc1-pc2_fa_with-meshed-lacunes.pdf/svg``

Colored by linear anisotropy:

- ``pc1-pc2_linear_anisotropy.svg``
- ``pc1-pc2_linear_anisotropy_with-meshed-lacunes.pdf/svg``

Colored by planar anisotropy:

- ``pc1-pc2_planar_anisotropy.svg``
- pc1-pc2_planar_anisotropy_with-meshed-lacunes.pdf/svg``

Colored by spherical anisotropy:

- ``pc1-pc2_spherical_anisotropy.svg``
- ``pc1-pc2_spherical_anisotropy_with-meshed-lacunes.svg/svg``

Methods
=======

PCA + clustering on 9 Invariant's moments
------------------------------------------

Compute PCA on 9 Invariant's moments (pure shape descriptors) exclude moment_invariant_5 and moment_invariant_9. Quote (Sun et al. 2007 MICCAI): *"we noticed that I6 and I10 were presenting bimodal distributions for some sulci. One mode was made up of positive values and the other one of negative values. There is no apparent correlation between the shape and the sign of I6 and I10... These 12 invariants denoted by I1, I2, ..., I12"*

Color by tensor's invariant to interpret findings
-------------------------------------------------

Tensor's invariant assume that the lacune can be modeled by an ellipsoide. Computed invariants (Ennis 2006):

- fractional anisotropy (FA)
- linear anisotropy
- planar anisotropy
- spherical anisotropy
- Mode: diffusion tensor mode

Use only linear and planar anisotropy of tensor's invariant
-----------------------------------------------------------

Plot lacunes in linear by planar anisotropy of tensor's invariant 2D space.

Results
=======

PCA + clustering on 9 Invariant's moments
------------------------------------------
The first two components explain 81% of the variance, the third component
explains 11%.

Here we plot the lacunes in the two first components of a PCA 9
Invariant's moments. To understand the discribution of the lacunes
in this shape's space, Lacunes were then colored with tensor's invariant
(FA, linear, planar and spherical anisotropy). Remember that those
**tensor's invariant were NEVER (yet) considered in the computation of the PCA.**

See directory ``results_moments_invariant``

Conclusions:

1. Invariant's moments capture the linear anisotropy (high top-left to low bottum right):

.. figure:: results_moments_invariant/pc1-pc2_linear_anisotropy_with-meshed-lacunes.png
	:scale: 200 %

	Two first components of a PCA on 9 Invariant's moments, colored by linear anisotropy. See file ``results_moments_invariant/pc1-pc2_linear_anisotropy[_with-meshed-lacunes].*``:


2. Invariant's moments capture (with some outliers) the planar anisotropy (low top-left to high bottum right):

.. figure:: results_moments_invariant/pc1-pc2_planar_anisotropy_with-meshed-lacunes.png
	:scale: 200 %

	Two first components of a PCA on 9 Invariant's moments, colored by planar anisotropy. See file ``results_moments_invariant/pc1-pc2_planar_anisotropy[_with-meshed-lacunes].*``: 


3. Lacunes' shape disctribution move from **(1) top-left**: high linear anisotropy (high FA) and low
planar anisotropy to **(2) middle**: lower linear anisotropy and lower planar anisotropy (low FA) to **(3) bottum right**: low linear anisotropy and high planar anisotropy (high FA):

.. figure:: results_moments_invariant/pc1-pc2_fa_with-meshed-lacunes.png
	:scale: 200 %

	Two first components of a PCA on 9 Invariant's moments, colored by fractionnal anisotropy. See file ``results_moments_invariant/pc1-pc2_planar_anisotropy[_with-meshed-lacunes].*``: 


results_tensor_invariant
------------------------
Material:
- pc1-pc2.pdf:  Lacunes ploted in the two first components, annotated with lacune_id and colored with tensor's invariant value.
- pca.txt: Explained variance ratio of the first two component is 99% + (PCA Loadings, weights vector).
- pca.csv: Explained variance ratio of the first two component is 99% + (PCA Loadings, weights vector).

Conclusion:
PCA capture two modes: linear anisotropy and spherical anisotropy

results_tensor_anisotropy_linear_spherical
------------------------------------------


Bibliography
============

- Fabrice Poupon PhD Thesis (in French), Sun et al. 20?? Automatic Inference of Sulcus Patterns Using 3D Moment Invariants, MICCAI??
- ZY. Sun, D. Rivière, F. Poupon, J. Régis, and J.-F. Mangin. Automatic inference of sulcus patterns using 3D moment invariants. In 10th Proc. MICCAI, LNCS Springer Verlag, pages 515-22, 2007
- Ennis DB, Kindlmann G. Orthogonal tensor invariants and the analysis of diffusion tensor magnetic resonance images. Magn Reson Med. 2006 Jan;55(1):136-46.

 
