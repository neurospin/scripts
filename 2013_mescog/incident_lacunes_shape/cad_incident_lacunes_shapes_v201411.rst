==========================================
Characterization of Incident Lacunes Shape
==========================================

:Author: Edouard.Duchesnay@cea.fr
:Date: 2014/10/27
:Build: ``rst2pdf cad_incident_lacunes_shapes_v201410.rst``


Characterization of incident lacunes shape using Invariant's moments and tensor's invariant.

.. contents::

Useful urls
===========

-**Data**:ftp://during@ftp.cea.fr/neuroimaging/CAD_incident_lacunes
    Restricted access limited to partners of MESCOG ERANET project.

-**Scripts**:https://github.com/neurospin/scripts/tree/master/2013_mescog/incident_lacunes_shape
    Restricted access limited NeuroSpin/UNATI/BrainOmics.


Shape descriptors
=================

**Ouputs**: all computed moments and descriptive statistics about the moments:
::

	incident_lacunes_moments.csv
	incident_lacunes_moments_descriptive.xls


=====================================   =========================================================
Name                                    Comment
=====================================   =========================================================
lacune_id                               lacune id
number_of_points                        -
vol_mm3                                 Volume in mm3 = M000*voxel volume 
order_1_monent_0                        M100 Normalized sum along x-axis
order_1_monent_1                        M010 Normalized sum along y-axis
order_1_monent_2                        M001 Normalized sum along z-axis
order_2_monent_0                        M200 (variance along x-axis)
order_2_monent_1                        M020 (variance along y-axis)
order_2_monent_2                        M002 (variance along z-axis)
order_2_monent_3                        M110 (co-variance x, y-axis)
order_2_monent_4                        M011 (co-variance y, z-axis)
order_2_monent_5                        M101 (co-variance x, z-axis)
order_3_monent_0                        \*
order_3_monent_1                        \*
order_3_monent_2                        \*
order_3_monent_3                        \*
order_3_monent_4                        \*
order_3_monent_5                        \*
order_3_monent_6                        \*
order_3_monent_7                        \*
order_3_monent_8                        \*
order_3_monent_9                        \*
                                        
center_of_mass_x                        Center of mass x-coordinate
center_of_mass_y                        Center of mass y-coordinate
center_of_mass_z                        Center of mass z-coordinate
                                        
orientation_v1_inertia                  Max inertia (v1)
orientation_v2_inertia                  Medium inertia (v2)
orientation_v3_inertia                  Min inertia (v3)
orientation_v1_x                        Max inertia (v1) x-orientation 
orientation_v1_y                        Max inertia (v1) y-orientation
orientation_v1_z                        Max inertia (v1) z-orientation
orientation_v2_x                        Medium inertia (v2) x-orientation
orientation_v2_y                        Medium inertia (v2) y-orientation
orientation_v2_z                        Medium inertia (v2) z-orientation
orientation_v3_x                        Min inertia (v3) x-orientation
orientation_v3_y                        Min inertia (v3) y-orientation
orientation_v3_z                        Min inertia (v3) z-orientation
                                        
moment_invariant_0                      moment invariant 0 \*
moment_invariant_1                      \*
moment_invariant_2                      \*
moment_invariant_3                      \*
moment_invariant_4                      \*
moment_invariant_5                      Do not consider this moment (Sun et al. MICCAI)
moment_invariant_6                      \*
moment_invariant_7                      \*
moment_invariant_8                      \*
moment_invariant_9                      Do not consider this moment (Sun et al. MICCAI)
moment_invariant_10                     \*
moment_invariant_11                     \*
                                        
area_mesh                               Lacune area calculated from a mesh
vol_mesh                                Lacune volume calculated from a mesh
perfo_orientation_x                     Perforator orientation
perfo_orientation_y                     Perforator orientation
perfo_orientation_z                     Perforator orientation
                                        
compactness                             vol_mesh^(2/3) / area_mesh

tensor_invariant_fa                     tensor invariant: Fractional Anisotropy
tensor_invariant_mode                   tensor invariant: mode
tensor_invariant_linear_anisotropy      tensor invariant: linear_anisotropy
tensor_invariant_planar_anisotropy      tensor invariant: planar_anisotropy
tensor_invariant_spherical_anisotropy   tensor invariant: spherical_anisotropy
perfo_angle_inertia_max                 angle between perforator and max inertia (v1) (in radian)
perfo_angle_inertia_min                 angle between perforator and min inertia (v3) (in radian)
=====================================   =========================================================

*See Fabrice Poupoun Thesis or ask Edouard.Duchesnay@cea.fr if needed. Below we
provide some details on computed moments:

- **Position-, scale-invariant moments (rotation- not invariant)**

    Order 1, 2 and 3 moments are centered and reduced moments  invariant in scale 
    and position but sensitive to rotation. Le Mpqr be a moment.
    Order 0 moment is such: p+q+r=0 the number of point in the object, noted M000
    Order 1 moments are such: p+q+r=1
    Order 2 moments are such: p+q+r=2
    Order 3 moments are such: p+q+r=3

    Order 1 moments:
    ::

        M100 : Sum(x) / M000
        M010 : Sum(y) / M000
        M001 : Sum(z) / M000


    Others moment general formula:
    Mpqr = Sum_x Sum_y Sum_z {(x-xc)^p (y-yc)^q (z-zc)^r p(x, y, z)}
    Where p(x, y, z) = 1 if (x, y, z) is in the object, 0 elsewhere.
    And xc is x coordinate of the the mass gravity point.

    Order 2 moments:
    ::

        u200 : Sum(x - xc)^2 / M000
        u110 : Sum(x - xc)(y - yc) / M000 ^ (puissance)
        ...


- **Position-invariant, rotation-invariant, and scale-invariant**

    See Fabrice Poupon PhD thesis 1999 (in French)


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
	--- mnts-inv_pca.csv
	--- mnts-inv_pca_descriptive.csv


	
	
Results
-------

Under subdirectory ``figures`` showing lacunes plotted in the **two first components** colored by FA, 
linear, planar and spherical anisotropy. File suffixed ``with-meshed-lacunes`` plot lacunes instead of simple dots.
File suffixed by ".svg" (Scalable Vector Graphics) are vectorial editable figures.

::


	results_moments_invariant/figures/
	# This file summarize most of the results in one pdf file
	--- mnts-inv_pc12.pdf

	# Color by tensors moment individual figures
	--- mnts-inv_pc12_fa.svg
	--- mnts-inv_pc12_fa_with-meshed-lacunes.pdf/svg

	--- mnts-inv_pc12_linear_anisotropy.svg
	--- mnts-inv_pc12_linear_anisotropy_with-meshed-lacunes.pdf/svg
		
	--- mnts-inv_pc12_planar_anisotropy.svg
	--- mnts-inv_pc12_planar_anisotropy_with-meshed-lacunes.pdf/svg
		
	--- mnts-inv_pc12_spherical_anisotropy.svg
	--- mnts-inv_pc12_spherical_anisotropy_with-meshed-lacunes.svg/svg
		
	--- mnts-inv_pc12_perfo_angle_inertia_max.svg/png
	--- mnts-inv_pc12_perfo_angle_inertia_min.svg/png

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

3. No visible link between the shape and maximum or minimum lacune orientation with the nearest perforator.

.. figure:: results_moments_invariant/figures/mnts-inv_pc12_perfo_angle_inertia_max.png
	:scale: 200 %

	Scatter plot of lacunes within the two first components of a PCA on 9 Invariant's moments, colored by the angle
	(radian in [0, PI/2]) formed by the maximim lacune orientation and the perforator.

.. figure:: results_moments_invariant/figures/mnts-inv_pc12_perfo_angle_inertia_min.png
	:scale: 200 %

	Scatter plot of lacunes within the two first components of a PCA on 9 Invariant's moments, colored by the angle
	(radian in [0, PI/2]) formed by the minimum lacune orientation and the perforator.


Directly use tensor's invariant
===============================

Methods
-------

1. Compute PCA on 5 Tensor's invariant

2. Use only linear and planar anisotropy of tensor's invariant


PCA components from tensor's invariant. It is a csv file of dimension: [n_lacunes x [lacune_id, PC1 (first component value), ..., PC5 (last component value)]]
And descriptive information about the PCA: explained variance ratio.
::

	results_tensor_invariant/
	--- tnsr-inv_pca.csv
	--- tnsr-inv_pca_descriptive.csv

Results
-------

1. Compute PCA on 5 Tensor's invariant and plot on the 2 first components.

PCA on 5 Tensor's invariant: PC1 capture the mode which demonstrate
that the main variability stem from change between planar anisotropic mode to linear anisotropic mode.
However this representation is not visually meaningful since it focuses
on few linear anisotropic lacunes.

::

	results_tensor_invariant/figures/
	--- tnsr-inv_pc12.pdf

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


.. figure:: results_tensor_invariant/figures/tnsr-inv_lin-plan_fa_with-meshed-lacunes_scaled.png
	:scale: 50 %

	Scatterplot of lacunes x-axis is linear anisotropy y-axis is planar, colored by fractionnal anisotropy.

No clear Relation between the shape, described with tensors moments
and the angle with the nearest perforator.

.. figure:: results_tensor_invariant/figures/tnsr-inv_lin-plan_perfo_angle_inertia_max.png
	:scale: 200 %

	Scatterplot of lacunes (x-axis is linear anisotropy, y-axis is planar),
	colored by angle formed by the main orientation axis and the perforator.

.. figure:: results_tensor_invariant/figures/tnsr-inv_lin-plan_perfo_angle_inertia_min.png
	:scale: 200 %

	Scatterplot of lacunes (x-axis is linear anisotropy, y-axis is planar),
	colored by angle formed by the smallest orientation axis (axis with smallest inertia) and the perforator.

Conclusion
==========

Most of the shape variability captured with moments' invariant could
be captured by linear and planar anisotropy which is based on an ellipsis modeling
of the shape of the lacunes.

No clear links with the nearest perforator could be found.


Material: mesh of lacunes
=========================

This directory contains GIfTI (meshs) files of lacunes that can be associated
with texture file to create nice 3D figures.

::

	results_lacunes-mesh/
	# Mesh of lacunes in PCA components (1 & 2) of moments invariants
	--- mnts-inv_pc12.gii
	--- mnts-inv_pc12_scaled.gii
	# Mesh of lacunes in linear, planar tensors invariant coordinates
	--- tnsr-inv_lin-plan.gii
	--- tnsr-inv_lin-plan_scaled.gii
	# Texture of lacunes us it to color lacunes mesh
	--- tex_lacune_id.gii
	--- tex_perfo_angle_inertia_max.gii
	--- tex_tensor_invariant_fa.gii
	--- tex_tensor_invariant_linear_anisotropy.gii
	--- tex_tensor_invariant_planar_anisotropy.gii
	--- tex_tensor_invariant_spherical_anisotropy.gii
    


Bibliography
============

- Fabrice Poupon PhD Thesis (in French), Sun et al. 20?? Automatic Inference of Sulcus Patterns Using 3D Moment Invariants, MICCAI??
- ZY. Sun, D. Rivière, F. Poupon, J. Régis, and J.-F. Mangin. Automatic inference of sulcus patterns using 3D moment invariants. In 10th Proc. MICCAI, LNCS Springer Verlag, pages 515-22, 2007
- Ennis DB, Kindlmann G. Orthogonal tensor invariants and the analysis of diffusion tensor magnetic resonance images. Magn Reson Med. 2006 Jan;55(1):136-46.



 
