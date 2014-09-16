==========================================
Characterization of Incident Lacunes Shape
==========================================

:Author: Edouard.Duchesnay@cea.fr
:Date: 2014/09/15
:Build: ``rst2pdfcad_incident_lacunes_shapes_v201409.rst``
:url: ftp://during@ftp.cea.fr/neuroimaging/CAD_incident_lacunes

Characterization of incident lacunes shape using Invariant's moments and tensor's invariant.

.. contents::

Shape descriptors
=================

**Ouputs**: all computed moments and descriptive statistics about the moments:
::

	incident_lacunes_moments.csv
	incident_lacunes_moments_descriptive.xls

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


Material: mesh of lacunes
=========================

::

	results_lacunes-mesh/
		# Mesh of lacunes in different coordinate space
		mnts-inv_pc12.gii
		mnts-inv_pc12_scaled.gii
		tnsr-inv_lin-plan.gii
		tnsr-inv_lin-plan_scaled.gii
		# Texture of lacunes
		tex_lacune_id.gii
		tex_perfo_angle_inertia_max.gii
		tex_tensor_invariant_fa.gii
		tex_tensor_invariant_linear_anisotropy.gii
		tex_tensor_invariant_planar_anisotropy.gii
		tex_tensor_invariant_spherical_anisotropy.gii


PCA on 10 invariant's moments
=============================	

Methods
-------

1. Compute PCA on 10 Invariant's moments (pure shape descriptors) exclude moment_invariant_5 and moment_invariant_9. Quote (Sun et al. 2007 MICCAI): *"we noticed that I6 and I10 were presenting bimodal distributions for some sulci. One mode was made up of positive values and the other one of negative values. There is no apparent correlation between the shape and the sign of I6 and I10... These 12 invariants denoted by I1, I2, ..., I12"*

2. Color by tensor's invariant to interpret findings

Tensor's invariant assume that the lacune can be modeled by an ellipsoid. Computed invariants (Ennis 2006):

- fractional anisotropy (FA)
- linear anisotropy
- planar anisotropy
- spherical anisotropy
- Mode: diffusion tensor mode

PCA components from moments' invariant. It is a csv file of dimension: [n_lacunes x [lacune_id, PC1 (first component value), ..., PC10 (last component value)]]
And descriptive information about the PCA: explained variance ratio of the first two components is 81% + (PCA Loadings, weights vector).
::

	results_moments_invariant/
		mnts-inv_pca.csv
		mnts-inv_pca_descriptive.csv

Under subdirectory ``figures`` showing lacunes plotted in the **two first components** colored by FA, 
linear, planar and spherical anisotropy. File suffixed ``with-meshed-lacunes`` plot lacunes instead of simple dots.

Many dot plot of lacunes (as dot) plotted in the two first components, 
annotated with lacune_id and colored with tensor's invariant value.
::

	results_moments_invariant/figures/mnts-inv_pc12.pdf
	
Results
-------

Colored by FA, linear, planar and spherical anisotropy:
::

	results_moments_invariant/figures/
		mnts-inv_pc12_fa.svg
		mnts-inv_pc12_fa_with-meshed-lacunes.pdf/svg
		
		mnts-inv_pc12_linear_anisotropy.svg
		mnts-inv_pc12_linear_anisotropy_with-meshed-lacunes.pdf/svg
		
		mmnts-inv_pc12_planar_anisotropy.svg
		mmnts-inv_pc12_planar_anisotropy_with-meshed-lacunes.pdf/svg
		
		mnts-inv_pc12_spherical_anisotropy.svg
		mnts-inv_pc12_spherical_anisotropy_with-meshed-lacunes.svg/svg
		
		mnts-inv_pc12_angle-with-perforator.png

The first two components explain 81% of the variance, the third component
explains 11%.

Here we plot the lacunes in the two first components of a PCA 9
Invariant's moments. To understand the distribution of the lacunes
in this shape's space, lacunes were then colored with tensor's invariant
(FA, linear, planar and spherical anisotropy). Remember that those
**tensor's invariant were NEVER (yet) considered in the computation of the PCA.**

Conclusions:

1. Invariant's moments capture the linear anisotropy (high top-left to low bottom right):

.. figure:: results_moments_invariant/figures/mnts-inv_pc12_linear_anisotropy_with-meshed-lacunes.png
	:scale: 200 %

	Scatter plot of lacunes within the two first components of a PCA on 9 Invariant's moments, colored by linear anisotropy.


2. Invariant's moments capture (with some outliers) the planar anisotropy (low top-left to high bottum right):

.. figure:: results_moments_invariant/figures/mnts-inv_pc12_planar_anisotropy_with-meshed-lacunes.png
	:scale: 200 %

	Scatter plot of lacunes within the two first components of a PCA on 9 Invariant's moments, colored by planar anisotropy.

3. Lacunes' shape distribution move from **(1) top-left**: high linear anisotropy (high FA) and low
planar anisotropy to **(2) middle**: lower linear anisotropy and lower planar anisotropy (low FA) 
to **(3) bottom right**: low linear anisotropy and high planar anisotropy (high FA):

.. figure:: results_moments_invariant/figures/mnts-inv_pc12_fa_with-meshed-lacunes.png
	:scale: 200 %

	Scatter plot of lacunes within the two first components of a PCA on 9 Invariant's moments, colored by fractional anisotropy.

3. No visible link between the shape and its orientation with the nearset perforator.

.. figure:: results_moments_invariant/figures/mnts-inv_pc12_angle-with-perforator.png

	Scatter plot of lacunes within the two first components of a PCA on 9 Invariant's moments, colored by the angle
	formed by the perforator and the main orientation of the lacune.


Directly use tensor's invariant
===============================

Methods
-------

1. Compute PCA on 5 Tensor's invariant


PCA components from tensor's invariant. It is a csv file of dimension: [n_lacunes x [lacune_id, PC1 (first component value), ..., PC5 (last component value)]]
And descriptive information about the PCA: explained variance ratio.
::

	results_tensor_invariant/
		tnsr-inv_pca.csv
		tnsr-inv_pca_descriptive.csv

Many dot plot of lacunes (as dot) ploted in the two first components, 
annotated with lacune_id and colored with tensor's invariant value.

::

	results_tensor_invariant/figures/
		tnsr-inv_pc12.pdf

2. Use only linear and planar anisotropy of tensor's invariant

Scatter plot of lacunes x-axis is linear anisotropy y-axis is planar.
File suffixed ``with-meshed-lacunes`` plot lacunes instead of simple dots.
File suffixed ``scaled`` plot lacunes whose dimension is scaled 
to the same global mean size.
::

	results_tensor_invariant/figures/
		tnsr-inv_lin-plan.pdf
		tnsr-inv_lin-plan_fa.svg
		tnsr-inv_lin-plan_fa_with-meshed-lacunes_noscaled.svg
		tnsr-inv_lin-plan_fa_with-meshed-lacunes_scaled.svg



Results
-------

PCA on 5 Tensor's invariant: PC1 capture the mode which demonstrate
that the main variability stem from change between planar anisotropic mode to linear anisotropic mode.
However this representation is not visually meaningful since it focuses
on few linear anisotropic lacunes.

.. figure:: results_tensor_invariant/figures/tnsr-inv_lin-plan_fa_with-meshed-lacunes_scaled.png
	:scale: 50 %

	Scatterplot of lacunes x-axis is linear anisotropy y-axis is planar, colored by fractionnal anisotropy.

No clear Relation with the perforator could be found.

.. figure:: results_tensor_invariant/figures/tnsr-inv_lin-plan_angle-with-perforator.png
	:scale: 200 %

	Scatterplot of lacunes (x-axis is linear anisotropy, y-axis is planar),
	colored by angle formed by the main orientation axis and the perforator.

Conclusion
==========

Most of the shape variability captured with moments' invariant could
be captured by linear and planar anisotropy which is based on an ellipsis modeling
of the shape of the lacunes.

No clear links with the nearest perforator could be found.


Bibliography
============

- Fabrice Poupon PhD Thesis (in French), Sun et al. 20?? Automatic Inference of Sulcus Patterns Using 3D Moment Invariants, MICCAI??
- ZY. Sun, D. Rivière, F. Poupon, J. Régis, and J.-F. Mangin. Automatic inference of sulcus patterns using 3D moment invariants. In 10th Proc. MICCAI, LNCS Springer Verlag, pages 515-22, 2007
- Ennis DB, Kindlmann G. Orthogonal tensor invariants and the analysis of diffusion tensor magnetic resonance images. Magn Reson Med. 2006 Jan;55(1):136-46.

 
