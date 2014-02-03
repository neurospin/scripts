require(ggplot2)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"
#setwd(WD)
#INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140110.csv", sep="/")
INPUT_MODELS = paste(BASE_DIR, "20140120_remove-predictors", sep="/")

source(paste(SRC,"utils.R",sep="/"))

setwd(INPUT_MODELS)

################################################################################################
## UTILS
################################################################################################
compare_models_stat<-function(yte1, ypred1, yte2, ypred2, w1.col.nonnull, w2.col.nonnull){
  inter =intersect(w1.col.nonnull, w2.col.nonnull)
  in1.not.in2 = setdiff(w1.col.nonnull, w2.col.nonnull)
  in2.not.in1 = setdiff(w2.col.nonnull, w1.col.nonnull)
  super2.nonnull = c(w1.col.nonnull, in2.not.in1) # 2 such 1 is nested in 2
  
  df1 = length(w1.col.nonnull) + 1
  df2 = length(super2.nonnull) + 1
  # QC all(Xyte1$y == Xyte2$y)  
  
  rsste1 = sum((yte1 - ypred1)^2)
  rsste2 = sum((yte2 - ypred2)^2)
  
  n = length(yte1)
  fstat = ((rsste1 - rsste2) / (df2 - df1)) / (rsste2 / (n - df1))
  
  pval=1-pf(fstat, df2 - df1, n - df2)
  
  return(data.frame(fstat=fstat, pval=pval, df1, df2, 
                    p1=length(w1.col.nonnull), p2=length(w2.col.nonnull), 
                    n.in1.not.in2=length(in1.not.in2), n.in2.not.in1=length(in2.not.in1), in2.not.in1=paste(in2.not.in1, collapse=",")))
}
#mod1_path="MDRS_TOTAL.M36~BASELINE+CLINIC.Rdata"; mod2_path="MDRS_TOTAL.M36~BASELINE+CLINIC+NIGLOB.Rdata"
compare_models<-function(mod1_path, mod2_path){
  #mod_path = mod1_path
  if(exists("result")) rm(result)
  load(mod1_path)
  r1 = result
  if(exists("result")) rm(result)
  load(mod2_path)
  r2 = result
  
  # FR > GE
  yte1 = r1$ygr
  ypred1 = r1$y_enet_pred_frgr
  yte2 = r2$ygr
  ypred2 = r2$y_enet_pred_frgr
  w1.col.nonnull = names(r1$enet_coef_fr)
  w2.col.nonnull = names(r2$enet_coef_fr)
  cmp_frge = data.frame(TEST="FR.GE", compare_models_stat(yte1, ypred1, yte2, ypred2, w1.col.nonnull, w2.col.nonnull))

  # FR > GE
  yte1 = r1$yfr[r1$fr_keep]
  ypred1 = r1$y_enet_pred_grfr
  yte2 = r2$yfr[r2$fr_keep]
  ypred2 = r2$y_enet_pred_grfr
  w1.col.nonnull = names(r1$enet_coef_gr)
  w2.col.nonnull = names(r2$enet_coef_gr)
  cmp_gefr = data.frame(TEST="GE.FR", compare_models_stat(yte1, ypred1, yte2, ypred2, w1.col.nonnull, w2.col.nonnull))
  
  cmp = rbind(cmp_frge, cmp_gefr)
  s1 = strsplit(mod1_path, "[~]")[[1]]
  s2 = strsplit(mod2_path, "[~]")[[1]]
  if(s1[1] != s2[1]){print("ERROR"); return(0)}
  mod1_str = strsplit(s1[2], "[.]")[[1]][1]
  mod2_str = strsplit(s2[2], "[.]")[[1]][1]
  cbind(data.frame(var=s1[1], PREDICTORS1=mod1_str, PREDICTORS2=mod2_str), cmp)
}

################################################################################################
## MODEL COMPARISON
################################################################################################

COMP=rbind(
compare_models(mod1_path="MDRS_TOTAL.M36~BASELINE+CLINIC.Rdata", mod2_path="MDRS_TOTAL.M36~BASELINE+CLINIC+NIGLOB.Rdata"),
compare_models(mod1_path="MMSE.M36~BASELINE+CLINIC.Rdata", mod2_path="MMSE.M36~BASELINE+CLINIC+NIGLOB.Rdata"),
compare_models(mod1_path="MRS.M36~BASELINE+CLINIC.Rdata", mod2_path="MRS.M36~BASELINE+CLINIC+NIGLOB.Rdata"),
compare_models(mod1_path="TMTB_TIME.M36~BASELINE+CLINIC.Rdata", mod2_path="TMTB_TIME.M36~BASELINE+CLINIC+NIGLOB.Rdata"))





var     PREDICTORS1            PREDICTORS2  TEST     fstat         pval df1 df2 p1 p2 n.in1.not.in2 n.in2.not.in1                 in2.not.in1
1 MDRS_TOTAL.M36 BASELINE+CLINIC BASELINE+CLINIC+NIGLOB FR.GE  2.403124 9.677821e-02   5   7  4  5             1             2                     LLV,BPF
2 MDRS_TOTAL.M36 BASELINE+CLINIC BASELINE+CLINIC+NIGLOB GE.FR -0.238565 1.000000e+00   6   9  5  6             2             3                LLV,WMHV,BPF
3       MMSE.M36 BASELINE+CLINIC BASELINE+CLINIC+NIGLOB FR.GE  5.043220 1.156882e-03   8  12  7  8             3             4 EDUCATION,TMTB_TIME,LLV,BPF
4       MMSE.M36 BASELINE+CLINIC BASELINE+CLINIC+NIGLOB GE.FR  5.253831 6.618982e-03   7   9  6  6             2             2                    LLV,WMHV
5        MRS.M36 BASELINE+CLINIC BASELINE+CLINIC+NIGLOB FR.GE  3.418778 2.152968e-02  11  14 10 10             3             3                LLV,WMHV,BPF
6        MRS.M36 BASELINE+CLINIC BASELINE+CLINIC+NIGLOB GE.FR 28.935151 4.376921e-07  14  15 13  6             8             1                         BPF
7  TMTB_TIME.M36 BASELINE+CLINIC BASELINE+CLINIC+NIGLOB FR.GE  5.462176 1.904113e-03  10  13  9  7             5             3             LLV,MBcount,BPF
8  TMTB_TIME.M36 BASELINE+CLINIC BASELINE+CLINIC+NIGLOB GE.FR -1.193550 1.000000e+00   6   8  5  6             1             2                WMHV,MBcount






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

