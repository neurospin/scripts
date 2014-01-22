#install.packages("glmnet")
require(glmnet)
require(ggplot2)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"
#setwd(WD)
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140110.csv", sep="/")
INPUT_MODELS = paste(BASE_DIR, "20140120_remove-predictors", sep="/")

source(paste(SRC,"utils.R",sep="/"))

setwd(INPUT_MODELS)

################################################################################################
## READ INPUT
################################################################################################
db = read_db(INPUT_DATA)
dim(db$DB_FR)# 239  42
dim(db$DB_GR)# 126  42
# remove normalized niglob variables

#mod1_path = "FR_ENET_MMSE.M36~BASELINE+CLINIC/all_bestcv_glmnet.Rdata"
#mod2_path = "FR_ENET_MMSE.M36~BASELINE+CLINIC+NIGLOB/all_bestcv_glmnet.Rdata"

# target = "MMSE.M36"
# dbtr = db$DB_FR
# dbte = db$DB_GR
db$col_targets
#[1] "TMTB_TIME.M36"  "MDRS_TOTAL.M36" "MRS.M36"        "BARTHEL.M36"    "MMSE.M36"  
baseline_clinic = rbind(
compare.models(mod1_path="FR_ENET_TMTB_TIME.M36~BASELINE+CLINIC/all_bestcv_glmnet.Rdata",
               mod2_path="FR_ENET_TMTB_TIME.M36~BASELINE+CLINIC+NIGLOB/all_bestcv_glmnet.Rdata",
               dbtr=db$DB_FR, dbte=db$DB_GR, target="TMTB_TIME.M36"),

compare.models(mod1_path="FR_ENET_MDRS_TOTAL.M36~BASELINE+CLINIC/all_bestcv_glmnet.Rdata",
               mod2_path="FR_ENET_MDRS_TOTAL.M36~BASELINE+CLINIC+NIGLOB/all_bestcv_glmnet.Rdata",
               dbtr=db$DB_FR, dbte=db$DB_GR, target="MDRS_TOTAL.M36"),

compare.models(mod1_path="FR_ENET_MRS.M36~BASELINE+CLINIC/all_bestcv_glmnet.Rdata",
               mod2_path="FR_ENET_MRS.M36~BASELINE+CLINIC+NIGLOB/all_bestcv_glmnet.Rdata",
               dbtr=db$DB_FR, dbte=db$DB_GR, target="MRS.M36"),

compare.models(mod1_path="FR_ENET_MMSE.M36~BASELINE+CLINIC/all_bestcv_glmnet.Rdata",
               mod2_path="FR_ENET_MMSE.M36~BASELINE+CLINIC+NIGLOB/all_bestcv_glmnet.Rdata",
               dbtr=db$DB_FR, dbte=db$DB_GR, target="MMSE.M36"))

# var                   fstat         pval df1 df2 p1 p2 n.in1.not.in2 n.in2.not.in1     in2.not.in1 R2.test.1 R2.test.2
# R2   TMTB_TIME.M36 4.219345 0.0082429545   9  12  8  7             4             3 LLV,MBcount,BPF      0.22      0.33
# R21 MDRS_TOTAL.M36 1.918935 0.1533762460   5   7  4  5             1             2         LLV,BPF      0.26      0.29
# R22        MRS.M36 7.357818 0.0011907556   9  11  8  9             1             2         LLV,BPF      0.36      0.46
# R23       MMSE.M36 9.366464 0.0002260948   8  10  7  6             3             2         LLV,BPF      0.37      0.49

baseline = rbind(
  compare.models(mod1_path="FR_GLM_TMTB_TIME.M36~BASELINE/all_glm.Rdata",
                 mod2_path="FR_ENET_TMTB_TIME.M36~BASELINE+NIGLOB/all_bestcv_glmnet.Rdata",
                 dbtr=db$DB_FR, dbte=db$DB_GR, target="TMTB_TIME.M36"),
  
  compare.models(mod1_path="FR_GLM_MDRS_TOTAL.M36~BASELINE/all_glm.Rdata",
                 mod2_path="FR_ENET_MDRS_TOTAL.M36~BASELINE+NIGLOB/all_bestcv_glmnet.Rdata",
                 dbtr=db$DB_FR, dbte=db$DB_GR, target="MDRS_TOTAL.M36"),
  
  compare.models(mod1_path="FR_GLM_MRS.M36~BASELINE/all_glm.Rdata",
                 mod2_path="FR_ENET_MRS.M36~BASELINE+NIGLOB/all_bestcv_glmnet.Rdata",
                 dbtr=db$DB_FR, dbte=db$DB_GR, target="MRS.M36"),
  
  compare.models(mod1_path="FR_GLM_MMSE.M36~BASELINE+CLINIC/all_glm.Rdata",
                 mod2_path="FR_ENET_MMSE.M36~BASELINE+CLINIC+NIGLOB/all_bestcv_glmnet.Rdata",
                 dbtr=db$DB_FR, dbte=db$DB_GR, target="MMSE.M36"))

#all_enet1$mod.glmnet$beta
#all_enet2$mod.glmnet$beta

