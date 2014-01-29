################################################################################################
## READ INPUT
################################################################################################
read_db=function(infile, to_remove = c()){#, to_remove = c("DELTA_BP", "TRIGLY", "MIGRAINE_WITH_AURA", "AVC", 
                         #                         "TRBEQUILIBRE", "TRBMARCHE", "DEMENTIA",
                         #                         "HYPERTENSION", "HYPERCHOL", "HDL", "FAST_GLUC", "NIHSS",
                         #                         "LLVn", "WMHVn", "BRAINVOL", "LLcount", "BARTHEL")){
  DB = read.csv(infile, header=TRUE, as.is=TRUE)
  col_info = c("ID", "SITE")
  col_targets =   c("TMTB_TIME.M36","MDRS_TOTAL.M36","MRS.M36","MMSE.M36")
  col_baselines =   c("TMTB_TIME","MDRS_TOTAL","MRS","MMSE")
  to_remove = unique(c(to_remove, col_info, col_targets, colnames(DB)[grep("36", colnames(DB))]))
  col_predictors = colnames(DB)[!(colnames(DB) %in% to_remove)]
  col_niglob = col_predictors[grep("LLV|LLcount|WMHV|MBcount|BPF", col_predictors)]
  col_clinic = col_predictors[!(col_predictors %in% col_niglob)]

  if(!all(sort(colnames(DB)) == sort(unique(c(col_info, col_targets, col_clinic, col_niglob, to_remove))))){
    print("ERROR COLNAMES DO NOT MATCH")
    sys.on.exit()
  }
  return(list(DB=DB, col_info=col_info,
              col_targets=col_targets, col_clinic=col_clinic, col_niglob=col_niglob, col_baselines=col_baselines))
}

################################################################################################
## MISSING DATA
################################################################################################


imput_missing<-function(D, skip=c()){
  # imput all missing values by colmean except M36
  #targets = colnames(D)[grep("M36",colnames(D))]
  predictors = colnames(D)[!(colnames(D) %in% skip)]
  Dimp = D
  imputation = NULL
  for(v in predictors){
    #v = "TMTB_TIME"
    for(i in which(is.na(D[,v]))){
      predictors_i = names(D[i, predictors])[!is.na(D[i, predictors]) & !(names(D[i, predictors]) %in% c("ID", "SITE"))]
      f_str = paste(v,paste(predictors_i, collapse="+"),sep="~")
      f = formula(f_str)
      mod = lm(f, D)
      Dimp[i, v] = predict(mod, D[i,,drop=F])
      imputation = rbind(imputation, data.frame(var=v, ID=D$ID[i], model=f_str))
    }
    #D[is.na(D[,v]), ] = mean(D[,v], na.rm=T)
  }
  return(list(Dimputed = Dimp, models=imputation))
}
