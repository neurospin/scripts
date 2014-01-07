## Response variables
## ==================

# response at t3, t2, t1
# --------------
scores.m36 = c("RANKIN_3", "TMTBT_3", "SCORETOT_3")
scores.m18 = c("RANKIN_2", "TMTBT_2", "SCORETOT_2")
scores.m0 = c("RANKIN_1", "TMTBT_1", "TOTAL_1")

# clinical cte
# ------------
#clinic.cte  = c("NCULTUREL_1", "SEXE.x", "TABAC", "CHOLHDL", "CHOLTOT",
#    "TRIGLY", "HOMOCY", "HBGLYCO","CRP", "CALCOOL")

#rm.var = c("CHOLHDL", "CHOLTOT", "TRIGLY")
# remove "CHOLHDL", "CHOLTOT", "TRIGLY" that have a strong center effect

demographic = c("NCULTUREL_1", "SEXE.x", "AGEIRM_1")
clinic.cte  = c("NCULTUREL_1", "SEXE.x", "TABAC", "HOMOCY", "HBGLYCO","CRP", "CALCOOL")

# clinic at t1
# ------------ 
clinic.m0   = c("AGEIRM_1", "PAS_1", "PAD_1")
clinic.m18   = c("AGEIRM_2", "PAS_2", "PAD_2")

# imagery global
#  -------------
## (time point, num == 1 || 2 || 3)
image.glob  = c("lacune_vol", "lesion_vol", "mb_num", "DWI_Peak","BPF")


# imagery sulci 
#  ------------

# (time point, num ==1 || 2 || 3 et par sillon)
image.sulci = c("geodesicDepthMean", "geodesicDepthMax", 
    "GM_thickness","fold_opening","surface.x")


var.m0 = c(scores.m0, clinic.cte, clinic.m0, 
    image.glob, image.sulci)


