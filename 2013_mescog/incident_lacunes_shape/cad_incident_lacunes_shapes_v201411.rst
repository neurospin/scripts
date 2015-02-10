==========================================
Characterization of Incident Lacunes Shape
==========================================

:Author: Edouard.Duchesnay@cea.fr
:Date: 2014/12/03
:Build: ``rst2pdf cad_incident_lacunes_shapes_v201411.rst``


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

**Scripts**:
::

    2013_mescog/incident_lacunes_shape/01_moments.py


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

Shape invariant's moments
-------------------------

See Fabrice Poupon Thesis or ask edouard.duchesnay@cea.fr if needed. Below we
provide some details on computed moments:

- **Position-, scale-invariant moments (rotation- not invariant)**

    Order 1, 2 and 3 moments are centered and reduced moments  invariant in scale 
    and position but sensitive to rotation. Let Mpqr be a moment.
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
        u110 : Sum(x - xc)(y - yc) / M000 ^ (power)
        ...


- **Position-invariant, rotation-invariant, and scale-invariant**

    See Fabrice Poupon PhD thesis 1999 (in French)


Tensor's invariant 
------------------
Those moments assume that the lacune can be modeled by an ellipsoid.
Computed invariants (see: Jolapara 2009):

- fractional anisotropy (FA)
- linear anisotropy (LA)
- planar anisotropy (PA)
- spherical anisotropy (SA)
- Mode: diffusion tensor mode (Mode) 

.. figure:: results_summary/scatter_matrix_tensors_inv_plus_angle.png
	:scale: 100 %

	Scatter plot of Tensor's invariant with the angle (in Degrees) formed by the perforator and the plan that contains most of the lacunes inertia ie.: the plan orthogonal to the orientation with minimum inertia.

Remark: For the calculation of spherical anisotropy (SA) we we followed the formula given (Jolapara 2009). Following this formula, this index is large when ellipsoid tend to be spherical. The formula makes sense but I would have call it spherical isotropy.


PCA on 10 invariant's moments
=============================	

Methods
-------

1. Compute PCA on 10 Invariant's moments (pure shape descriptors) exclude moment_invariant_5 and moment_invariant_9. Quote (Sun et al. 2007 MICCAI): *"we noticed that I6 and I10 were presenting bimodal distributions for some sulci. One mode was made up of positive values and the other one of negative values. There is no apparent correlation between the shape and the sign of I6 and I10... These 12 invariants denoted by I1, I2, ..., I12"*

2. Color by tensor's invariant to interpret findings

PCA components from moments' invariant. It is a csv file of dimension: [n_lacunes x [lacune_id, PC1 (first component value), ..., PC10 (last component value)]]
And descriptive information about the PCA: explained variance ratio of the first two components is 81% + (PCA Loadings, weights vector).
	
	
Results
-------

Under sub-directory ``figures`` showing lacunes plotted in the **two first components** colored by FA, 
linear, planar and spherical anisotropy. File suffixed ``with-meshed-lacunes`` plot lacunes instead of simple dots.
File suffixed by ".svg" (Scalable Vector Graphics) are vectorial editable figures.

::

	results_moments_invariant/
    # Components
	--- mnts-inv_pca.csv
	--- mnts-inv_pca_descriptive.csv

	results_moments_invariant/figures/
	# Color by tensors moment individual figures
	--- mnts-inv_pc12.pdf           # This pdf file contains most of plots
	--- mnts-inv_pc12_fa.png
	--- mnts-inv_pc12_fa.svg
	--- mnts-inv_pc12_linear_anisotropy.png
	--- mnts-inv_pc12_linear_anisotropy.svg
	--- mnts-inv_pc12_perfo_angle_inertia_max.png
	--- mnts-inv_pc12_perfo_angle_inertia_max.svg
	--- mnts-inv_pc12_perfo_angle_inertia_min.png
	--- mnts-inv_pc12_perfo_angle_inertia_min.svg
	--- mnts-inv_pc12_planar_anisotropy.png
	--- mnts-inv_pc12_planar_anisotropy.svg
	--- mnts-inv_pc12_spherical_anisotropy.png
	--- mnts-inv_pc12_spherical_anisotropy.svg

The first two components explain 81% of the variance, the third component
explains 11%.

Here we plot the lacunes in the two first components of a PCA 9
Invariant's moments. To understand the distribution of the lacunes
in this shape's space, lacunes were then colored with tensor's invariant
(FA, linear, planar and spherical anisotropy). Remember that those
**tensor's invariant were NEVER (yet) considered in the computation of the PCA.**

Conclusions:

1. Invariant's moments capture the tensors anisotropies (high top-left to low bottom right):

.. figure:: results_moments_invariant/figures/mnts-inv_pc12_anisotropies.png
	:scale: 100 %

	Scatter plot of lacunes within the two first components of a PCA on 9 Invariant's moments, colored by FA, linear anisotropy, planar anisotropy and spherical anisotropy.


.. figure:: results_moments_invariant/figures/mnts-inv_pc12_mesh.png
	:scale: 100 %

	Scatter plot of meshed lacunes within the two first components of a PCA on 9 Invariant's moments, colored by FA. Left: maximum inertia orientation is aligned to y-axis. Right: perforator is aligned to y-axis and residual maximum inertia in (x-z plan) is aligned to x-axis.


2. Orientation with maximum inertia is aligned with the nearest perforator

.. figure:: results_moments_invariant/figures/mnts-inv_pc12_angle-with-perforator.png
	:scale: 100 %

	Scatter plot of lacunes within the two first components of a PCA on 9 Invariant's moments, colored by the angle (radian in [0, PI/2]) formed by the (left) maximum / (right) minimum lacune orientation and the perforator. Left figure shows smaller angles and thus better alignment.


PCA on tensor's invariant
=========================

Methods
-------

Compute PCA on 4 Tensor's invariant (do not use the mode)

.. 2. Use only linear and planar anisotropy of tensor's invariant


PCA components from tensor's invariant. It is a csv file of dimension: [n_lacunes x [lacune_id, PC01 (first component value), ..., PC04 (last component value)]]
And descriptive information about the PCA: explained variance ratio.

Results
-------

Under sub-directory ``figures`` showing lacunes plotted in the **two first components** 
(file with ``pc12``) or directly in linear-planar plan (files with ``lin-plan``).
Dots are colored by FA, linear, planar and spherical anisotropy. Files with ``mesh`` 
plot lacunes (from anatomist) instead of simple dots.
Files suffixed by ".svg" (Scalable Vector Graphics) are vectorial editable figures.

::

    results_tensor_invariant/
    # Components
	--- tnsr-inv_pca.csv
	--- tnsr-inv_pca_descriptive.csv

    results_moments_invariant/figures
    # some conlcusion image
	--- mnt-inv_tnsr-inv_pc12_fa_meshs_illustrated.png

    # In linear-planar plan:
	--- tnsr-inv_lin-plan.pdf          # This pdf file contains most of plots
	--- tnsr-inv_lin-plan_perfo_angle_inertia_max.png
	--- tnsr-inv_lin-plan_perfo_angle_inertia_max.svg
	--- tnsr-inv_lin-plan_perfo_angle_inertia_min.png
	--- tnsr-inv_lin-plan_perfo_angle_inertia_min.svg

    # In two first components:
	--- tnsr-inv_pc12.pdf              # This pdf file contains most of plots
	--- tnsr-inv_lin-plan_fa.png
	--- tnsr-inv_lin-plan_fa.svg
	--- tnsr-inv_pc12_angles-with-perforator.png
	--- tnsr-inv_pc12_anisotropies.png
	--- tnsr-inv_pc12_fa__max_inertia_to_yaxis-mesh.png
	--- tnsr-inv_pc12_fa_meshs_illustrated.png
	--- tnsr-inv_pc12_fa_meshs.png
	--- tnsr-inv_pc12_fa__perfo_to_yaxis-mesh.png
	--- tnsr-inv_pc12_fa.png
	--- tnsr-inv_pc12_fa.svg
	--- tnsr-inv_pc12_figures.odg
	--- tnsr-inv_pc12_figures.odg
	--- tnsr-inv_pc12_linear_anisotropy.png
	--- tnsr-inv_pc12_linear_anisotropy.svg
	--- tnsr-inv_pc12_perfo_angle_inertia_max.png
	--- tnsr-inv_pc12_perfo_angle_inertia_max.svg
	--- tnsr-inv_pc12_perfo_angle_inertia_min.png
	--- tnsr-inv_pc12_perfo_angle_inertia_min.svg
	--- tnsr-inv_pc12_planar_anisotropy.png
	--- tnsr-inv_pc12_planar_anisotropy.svg
	--- tnsr-inv_pc12_spherical_anisotropy.png
	--- tnsr-inv_pc12_spherical_anisotropy.svg

1. PCA on Tensors' invariants capture a most and a meaningful variability of the
lacunes' shape:

The first two components explain 99% of the variance. Most of the variability
is due variations along 3 poles: top-left low anisotropy, top right
linear anisotropy and bottom planar anisotropy.

.. figure:: results_tensor_invariant/figures/tnsr-inv_pc12_anisotropies.png
	:scale: 100 %

	Scatter plot of lacunes within the two first components of a PCA on 4 Tensors' invariants, colored by FA, linear anisotropy, planar anisotropy and spherical anisotropy.


.. figure:: results_tensor_invariant/figures/tnsr-inv_pc12_fa_meshs.png
	:scale: 100 %

	Scatter plot of meshed lacunes within the two first components of a PCA on 4 Tensors' invariants, colored by FA. Left: maximum inertia orientation is aligned to y-axis. Right: perforator is aligned to y-axis and residual maximum inertia in (x-z plan) is aligned to x-axis.


.. figure:: results_tensor_invariant/figures/tnsr-inv_pc12_angles-with-perforator.png
	:scale: 100 %

	Scatter plot of lacunes within the two first components of a PCA on 4 Tensors' invariants, colored by the angle (radian in [0, PI/2]) formed by the (left) maximum / (right) minimum lacune orientation and the perforator. Left figure shows smaller angles and thus better alignment.

..
    ############################################################################
    ## COMMENTED
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
    ############################################################################


Conclusion
==========

Moments' invariant are hypothesis free shape descriptors. We demonstrate that the capture variability correspond to the variability that could be captured by tensor's invariant.
The lacunes' shape follow a continuum between three poles:  low anisotropy "spheres"
linear anisotropy "cigars" and planar anisotropy "saucer".


.. figure:: results_summary/conclusion.png
	:scale: 100 %

	Scatter plot of lacunes within the two first components of:Left: PCA on 9 Invariant's moments and right: PCA on 4 Tensors' invariants. Lacunes mesh are colored by FA. Lacune were rotated such that the maximum inertia orientation is aligned to y-axis.


The lacunes are aligned with the nearest perforator:

.. figure:: results_summary/angle_deg_perforator_plan_orthogonal_to_orientation_with_min_inertia.png


    Histogram (in Degrees) of the angle formed by the perforator and the plan that contains most of the lacunes inertia ie.: the plan orthogonal to the orientation with minimum inertia.


Material: mesh of lacunes
=========================

This directory contains GIfTI (meshs) files of lacunes that can be associated
with texture file to create nice 3D figures. Naming convention:

<space>__<object>__<orientation>_<scaling>

space:
    - ``brain``: brain space, **It is wrong !!** since native spaces are not the same. Ask Benno.

    - ``mnts_inv_pc12``: In space defined by the two first components of PCA on  9 Invariant's moments.

    - ``tnsr_inv_lin_plan``: In space defined by the linear and planar anisotropy of tensor's invariant.

    - ``tnsr_inv_pc12``: In space defined by the two first components of PCA on the 4 Tensor's invariant.

object:
    - ``lacunes``: Mesh of lacunes.

    - ``perfo``: Mesh of perforators.

orientation:
    - ``native``: No rotation.

    - ``max_inertia_to_yaxis``: Objects were rotated such that the maximum inertia orientation is aligned to y-axis.

    - ``perfo_to_yaxis``: Objects were rotated such that the nearest perforator is aligned to y-axis. Then, Object were turned around the perforator such that the maximum residual inertia, in x-z plan, is aligned to x-axis.

scaling:
    - ``scaled``: Lacunes sizes were globally scaled such that they (approximately) all have the same volume.


``lacunes`` ``perfo`` define the meshed object.


::

    results_lacunes-mesh/
    # In brain space, Wrong !! since native spaces are not the same
	--- brain__lacunes__native.gii
	--- brain__perforators__native.gii

    # In space defined by the two first components of PCA on 9 Invariant's moments
	--- mnts_inv_pc12__lacunes__max_inertia_to_yaxis.gii
	--- mnts_inv_pc12__lacunes__max_inertia_to_yaxis_scaled.gii
	--- mnts_inv_pc12__lacunes__perfo_to_yaxis.gii
	--- mnts_inv_pc12__lacunes__perfo_to_yaxis_scaled.gii
	--- mnts_inv_pc12__perfo__max_inertia_to_yaxis.gii

    # In space defined by the linear and planar anisotropy of tensor's invariant
	--- tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis.gii
	--- tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_scaled.gii
	--- tnsr_inv_lin_plan__lacunes__perfo_to_yaxis.gii
	--- tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_scaled.gii

    # In space defined by the two first components of PCA on the 4 Tensor's invariant
	--- tnsr_inv_pc12__lacunes__max_inertia_to_yaxis.gii
	--- tnsr_inv_pc12__lacunes__max_inertia_to_yaxis_scale.gii
	--- tnsr_inv_pc12__lacunes__perfo_to_yaxis.gii
	--- tnsr_inv_pc12__lacunes__perfo_to_yaxis_scaled.gii
	--- tnsr_inv_pc12__perfo__max_inertia_to_yaxis.gii

    # Textures for lacunes or perforators
	--- tex__lacunes__lacune_id.gii
	--- tex__lacunes__perfo_angle_inertia_max.gii
	--- tex__lacunes__tensor_invariant_fa.gii
	--- tex__lacunes__tensor_invariant_linear_anisotropy.gii
	--- tex__lacunes__tensor_invariant_planar_anisotropy.gii
	--- tex__lacunes__tensor_invariant_spherical_anisotropy.gii
	--- tex__perforators__lacune_id.gii
	--- tex__perforators__perfo_angle_inertia_max.gii
	--- tex__perforators__tensor_invariant_fa.gii
	--- tex__perforators__tensor_invariant_linear_anisotropy.gii
	--- tex__perforators__tensor_invariant_planar_anisotropy.gii
	--- tex__perforators__tensor_invariant_spherical_anisotropy.gii
  

Anatomist usage:

1. Load all mesh: ``anatomist results_lacunes-mesh/*.gii &``

2. Fusion laucune mesh with one texture example: ``mnts_inv_pc12__lacunes__perfo_to_yaxis_scaled.gii`` with ``tex__lacunes__tensor_invariant_fa.gii``

3. Drop the Fusion object ``TEXTURED SURF...`` in the ``3D`` view, click on axial view.

4. Fusion perforators mesh with one texture example: ``mnts_inv_pc12__perfo__max_inertia_to_yaxis.gii`` with ``tex__perforators__tensor_invariant_fa.gii``

5. Drop the second Fusion object (perforators) in the first windows to superimpose the two in the same windows.
 
If the texture do not work,  right click Fusion object ``TEXTURED SURF...``
Color / Texturing, click something (eg. linear - object) the re-click None,
to force the texturing

Y-axis is flipped compared to dot scatter plot. This is caused by the convention
used by anatomist. Y moves top -> down. Do not worry, save the image an flip it
with your favorite software.


Bibliography
============

- Fabrice Poupon, PhD Thesis (in French)

- ZY. Sun, D. Rivière, F. Poupon, J. Régis, and J.-F. Mangin. Automatic inference of sulcus patterns using 3D moment invariants. In 10th Proc. MICCAI, LNCS Springer Verlag, pages 515-22, 2007

- Ennis DB, Kindlmann G. Orthogonal tensor invariants and the analysis of diffusion tensor magnetic resonance images. Magn Reson Med. 2006 Jan;55(1):136-46.

- Jolapara et al. Diffusion tensor mode in imaging of intracranial epidermoid cysts: one step ahead of fractional anisotropy,  Neuroradiology (2009) 51:123–129




 
