import os

path_areals = "/volatile/Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k"
output_dir = "/neurospin/brainomics/2016_HCP/functional_analysis/HCP_MMP1.0"
sides = ['L', 'R']
dir_resample_by_HCP_fsaverage = "/volatile/Pipelines/global/templates/standard_mesh_atlases/resample_fsaverage/"


for side in sides:
    file_areals = os.path.join(path_areals, "Q1-Q6_RelatedParcellation210."+side+".CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii")
    file_gii = os.path.join(output_dir, "Q1-Q6_RelatedParcellation210."+side+".CorticalAreas_dil_Colors.32k_fs_LR.label.gii")
    cii_midthick_41k = os.path.join(dir_resample_by_HCP_fsaverage,"fsaverage6."+side+".midthickness_va_avg.41k_fsavg_"+side+".shape.gii")
    gii_midthick_41k = os.path.join(dir_resample_by_HCP_fsaverage,"fsaverage6."+side+".midthickness_va_avg.41k_fsavg_"+side+".surf.gii")
    if side == "L":
        cmd = "wb_command -cifti-separate "+file_areals+" COLUMN -label CORTEX_LEFT "+file_gii
        cmd0 = "wb_command -cifti-separate "+cii_midthick_41k+" COLUMN -label CORTEX_LEFT "+gii_midthick_41k
    else:
        cmd = "wb_command -cifti-separate "+file_areals+" COLUMN -label CORTEX_RIGHT "+file_gii
        cmd0 = "wb_command -cifti-separate "+cii_midthick_41k+" COLUMN -label CORTEX_LEFT "+gii_midthick_41k
    print cmd
    os.system(cmd)
    metric_in = file_gii
    metric_out = os.path.join(output_dir, side+".fsaverage41k.label.gii")
    midthick_reg0  = os.path.join(dir_resample_by_HCP_fsaverage, "fs_LR."+side+".midthickness_va_avg.32k_fs_LR.shape.gii")
    fs_LR_to_32k = os.path.join(dir_resample_by_HCP_fsaverage, "fs_LR-deformed_to-fsaverage."+side+".sphere.32k_fs_LR.surf.gii")
    fsavg_std_41k = os.path.join(dir_resample_by_HCP_fsaverage,"fsaverage6_std_sphere."+side+".41k_fsavg_"+side+".surf.gii")
    cmd = "wb_command -label-resample "+metric_in+" "+fs_LR_to_32k+" "+fsavg_std_41k+" ADAP_BARY_AREA "+metric_out+" -area-metrics "+midthick_reg0+" "+cii_midthick_41k    
    print cmd
    os.system(cmd)
