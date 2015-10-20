require(ggplot2)
library(RColorBrewer)
require(glmnet)
library(rpart)
library(ipred) # install.packages("ipred")
library(XLConnect) # install.packages("XLConnect")
# read
#wb = loadWorkbook(filename)
#data1 = readWorksheet(wb,sheet1,...)
#data2 = readWorksheet(wb,sheet2,...)
# write
#wb = loadWorkbook(filename)
#createSheet(wb, sheet1)
#writeWorksheet(wb, data1, sheet1, ...)
#saveWorkbook(wb)

#library(RColorBrewer)
#library(plyr)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"
source(paste(SRC,"utils.R",sep="/"))

# INPUT ---
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140728_nomissing_BPF-LLV_imputed.csv", sep="/")

# OUTPUT ---
#OUTPUT = paste(BASE_DIR, sub(".csv", "", sub("dataset_clinic_niglob_", "", basename(INPUT_DATA))), "enet", "M36", sep="/")
OUTPUT = paste(BASE_DIR, sub(".csv", "", sub("dataset_clinic_niglob_", "", basename(INPUT_DATA))), "2015-10", sep="/")

if (!file.exists(OUTPUT)) dir.create(OUTPUT)
VALIDATION = "BOOT"
#VALIDATION = "All"
#VALIDATION = "FR-GE"
RM_TEST_OUTLIERS = FALSE

source(paste(SRC,"utils.R",sep="/"))

# rsync -azvun --delete /neurospin/mescog/proj_predict_cog_decline ~/data/
# rsync -azvun --delete  ~/data/proj_predict_cog_decline /neurospin/mescog/

################################################################################################
## READ INPUT
################################################################################################
db = read_db(INPUT_DATA)
#db$DB

db$DB$TMTB_TIME.CHANGE = (db$DB$TMTB_TIME.M36 - db$DB$TMTB_TIME)
db$DB$MDRS_TOTAL.CHANGE = (db$DB$MDRS_TOTAL.M36 - db$DB$MDRS_TOTAL)
db$DB$MRS.CHANGE = (db$DB$MRS.M36 - db$DB$MRS)
db$DB$MMSE.CHANGE = (db$DB$MMSE.M36 - db$DB$MMSE)

PNZERO = 0.5# c(.1, .25 , .5, .75)
ALPHA = 0.25# seq(0, 1, .25)
#SEEDS = 46#seq(1, 100)
TARGETS = c("TMTB_TIME.CHANGE", "MDRS_TOTAL.CHANGE", "MRS.CHANGE", "MMSE.CHANGE")
BASELINES = unlist(lapply(strsplit(TARGETS, "[.]"), function(x)x[1]))

NBOOT = 200


SETTINGS = list("BASELINE"       = c(),
                "BASELINE+NIGLOB"       = db$col_niglob,
                "BASELINE+NIGLOBFULL"       = c(db$col_niglob, db$col_niglob_full),
                "BASELINE+CLINIC"       = c(BASELINES, db$col_clinic),
                "BASELINE+CLINIC+NIGLOB"= c(BASELINES, db$col_clinic, db$col_niglob),
                "BASELINE+CLINIC+NIGLOBFULL"= c(BASELINES, db$col_clinic, db$col_niglob, db$col_niglob_full))

RESULTS_TAB_ALL = NULL
RESULTS_TAB_MEAN = NULL
COMPARISONS = NULL

RESULTS = list()

PREDICTORS_FULL = SETTINGS[["BASELINE+CLINIC+NIGLOBFULL"]]#c(BASELINES, )

#####################################################################################################
## Models

## simple
fitpredict_simple<-function(X, y, mod="learn", model=NULL){
  # target = "MMSE.CHANGE"
  # target = "MRS.CHANGE"
  # target = "TMTB_TIME.CHANGE"
  # target = "MDRS_TOTAL.CHANGE"
  # model = simple_prediction(target, d=db$DB, mod="learn")
  # simple_prediction(target, d=db$DB, mod="predict", model=model)
  if(TARGET == "MMSE.CHANGE"){
    m1 = X[,"LLV"]>=1964                                                                                 # decline
    m2 = (X[,"LLV"]< 1964 & X[,"MMSE"]>=27.5) | (X[,"LLV"]<1964 & X[,"MMSE"]< 27.5 & X[,"BPF"]< 0.7645)  # stable
    m3 = (X[,"LLV"]< 1964 & X[,"MMSE"]< 27.5 & X[,"BPF"]>=0.7645)                                        # improve
  }
  if(TARGET == "MRS.CHANGE"){
    m3 = (X[,"MRS"]>=1.5) & (X[,"LLV"]< 1251)        # improve
    m2 = (X[,"MRS"]< 1.5) & (X[,"BPF"]>=0.8572)      # stable
    m1 = !(m3 | m2)                                  # decline
  }
  if(TARGET == "TMTB_TIME.CHANGE"){
    m3 = X[,"TMTB_TIME"]>=173                        # improve
    m2 = (X[,"TMTB_TIME"]<173) & (X[,"LLV"]<394.9)   # stable
    m1 = (X[,"TMTB_TIME"]<173) & (X[,"LLV"]>=394.9)  # decline
  }
  if(TARGET == "MDRS_TOTAL.CHANGE"){
    m1 = (X[,"LLV"]>=1632) | (X[,"LLV"]<1632 & X[,"BPF"]<0.749) # decline
    m2 = !m1                                                    # stable
    m3 = !(m1 | m2)
  }
  if(sum(m1 | m2 | m3) != dim(X)[1]){
    print("Error")
  }
  if(mod == "learn"){
    grp_mean = c(mean(y[m1], na.rm=TRUE), mean(y[m2], na.rm=TRUE), mean(y[m3], na.rm=TRUE))
    labels = rep(NA, dim(X)[1])
    labels[m1] = "decline"
    labels[m2] = "stable"
    labels[m3] = "improve"
    return(list(grp_mean=grp_mean, labels=labels))
  }
  pred = rep(NA, dim(X)[1])
  pred[m1] = model$grp_mean[1]
  pred[m2] = model$grp_mean[2]
  pred[m3] = model$grp_mean[3]
  return(pred)
}

fit_simple<-function(x, y){
  return(fitpredict_simple(x, y, mod="learn", model=NULL))
}

predict_simple<-function(model, x){
  return(fitpredict_simple(x, y, mod="predict", model=model))
}
# m = fit_simple(X, y)
# predict_simple(m, X)

## enet
fit_enet<-function(X, y){
  cv_glmnet = cv.glmnet(X, y, alpha=ALPHA)
  lambda = cv_glmnet$lambda.min # == cv_glmnet$lambda[which.min(cv_glmnet$cvm)]
  enet_nzero = cv_glmnet$nzero[which.min(cv_glmnet$cvm)][[1]]
  enet_nzero_min = max(round(dim(X)[2]*PNZERO), 2)
  if(enet_nzero < enet_nzero_min)
    lambda = cv_glmnet$lambda[which(cv_glmnet$nzero > enet_nzero_min)[1]]
  # if cannot find such lambda take the last one (least penalization)
  if(is.na(lambda)) lambda = cv_glmnet$lambda[length(cv_glmnet$lambda)]
  mod_enet = glmnet(X, y, lambda=lambda, alpha=ALPHA)
}
predict_enet<-function(model, x){
  return(predict(model, x))
}

fit_glm <- function(x, y){lsfit(x,y)}
predict_glm <- function(fit,x){
  cbind(1,x)%*%fit$coef
}

#####################################################################################################
## 632 bootstrap + ci
#install.packages('bootstrap')
#library(bootstrap)
# 
# ci => boott
# ex:
# x <- rnorm(100, mean=-3)
#theta <- function(x){mean(x)}
#boott(x,theta)

# ci
ci<-function(theta_star, perc=c(.001,.01,.025,.05,.10,.25,.50,.75,.90,.95,.975,.99,.999)){
  ans = matrix(NA, nrow=ncol(theta_star), ncol=length(perc), dimnames=list(colnames(theta_star), as.character(perc)))
  for(theta_star_name in colnames(theta_star)){
    val_sorted = sort(theta_star[, theta_star_name])#[length(theta_star):1]
    o <- trunc(length(val_sorted) * perc) + 1
    ans[theta_star_name, ] = val_sorted[o]
  }
  return(ans)
}

bootpred_ci <-function(x, y, nboot, mod_fit, mod_predict, theta, ...){
  #x=X;y=y; nboot=NBOOT; mod_fit=fit_enet; mod_predict=predict_enet; theta=losses
  #x=X; y=y; nboot=NBOOT; mod_fit=fit_glm; mod_predict=predict_glm; theta=losses;
  x <- as.matrix(x)
  n <- length(y)
  fit0 <- mod_fit(x, y)
  yhat_0 <- mod_predict(fit0, x)
  theta_0 <- theta(y, yhat_0)
  theta_star = matrix(NA, nrow=nboot, ncol=length(theta_0), dimnames=list(NULL, names(theta_0)))
  yhat_star = matrix(NA, nrow=nboot, ncol=length(y))
  yhat_632 = matrix(NA, nrow=nboot, ncol=length(y))
  for (b in 1:nboot) {
    set.seed(b)
    ii <- sample(1:n, replace = TRUE)
    fit <- mod_fit(x[ii, ], y[ii])
    y_hat = mod_predict(fit, x[-ii, , drop=FALSE])
    theta_star[b, ] = theta(y[-ii], y_hat)
    yhat_star[b, -ii] = y_hat
    yhat_632[b, -ii] = 0.368 * yhat_0[-ii] + 0.632 * y_hat
  }
  
  theta_star_mu = apply(theta_star, 2, mean)
  theta_star_sd = apply(theta_star, 2, sd)
  theta_632_mu = 0.368 * theta_0 + 0.632 * theta_star_mu
  theta_632_sd = 0.632 * theta_star_sd

  theta_star_mu_ci = ci(theta_star)# boott(theta_star, theta=mean, nboott = 200)$confpoints
  theta_star_632 = 0.368 * theta_0 + 0.632 * theta_star
  theta_632_mu_ci = ci(theta_star_632) # boott(theta_star_632, theta=mean, nboott = 200)$confpoints
  #ci(theta_star)
  return(list(theta_star_mu=theta_star_mu, theta_star_sd=theta_star_sd, theta_632_mu=theta_632_mu, theta_632_sd=theta_632_sd, 
              theta_star_ci=theta_star_mu_ci, theta_632_ci=theta_632_mu_ci, yhat_star=yhat_star, yhat_632=yhat_632,
              theta_star=theta_star, theta_star_632=theta_star_632))
}

square.err <- function(y, yhat) {c(mse=(y-yhat)^2)}

mse <- function(y, yhat) {c(mse=mean((y-yhat)^2))}

r2<-function(y_true, y_pred){
  ## r2: http://en.wikipedia.org/wiki/Coefficient_of_determination
  df1=length(y_true)
  SS.tot       = sum((y_true - mean(y_true))^2)
  #SS.tot.unbiased     = sum((y_true - mean(y.train))^2)
  SS.err       = sum((y_true - y_pred)^2)
  mse = SS.err/df1
  r2  = 1 - SS.err/SS.tot
  return(c(r2=r2))
}

losses<-function(y, yhat){
  return(c(mse(y, yhat), r2(y, yhat)))
}

##############################################################################################################
## MISC UTILS
format_tuple<-function(target, predictors, model, res, bic, coef){
  if(is.null(res))return(NULL)
  cbind(
    data.frame(
      target    =target,
      predictors=predictors,
      model     =model,
      
      mse       =res$theta_star_mu[["mse"]],
      mse_se    =res$theta_star_sd[["mse"]],
      mse_ci.025=res$theta_star_ci["mse", "0.025"],
      mse_ci.25=res$theta_star_ci["mse", "0.25"],
      mse_ci.75=res$theta_star_ci["mse", "0.75"],
      mse_ci.975=res$theta_star_ci["mse", "0.975"],
      
      r2       =res$theta_star_mu[["r2"]],
      r2_se    =res$theta_star_sd[["r2"]],
      r2_ci.025=res$theta_star_ci["r2", "0.025"],
      r2_ci.25 =res$theta_star_ci["r2", "0.25"],
      r2_ci.75 =res$theta_star_ci["r2", "0.75"],
      r2_ci.975=res$theta_star_ci["r2", "0.975"],
      
      mse_632       =res$theta_632_mu[["mse"]],
      mse_632_se    =res$theta_632_sd[["mse"]],
      mse_632_ci.025=res$theta_632_ci["mse", "0.025"],
      mse_632_ci.25 =res$theta_632_ci["mse", "0.25"],
      mse_632_ci.75 =res$theta_632_ci["mse", "0.75"],
      mse_632_ci.975=res$theta_632_ci["mse", "0.975"],
      
      r2_632       =res$theta_632_mu[["r2"]],
      r2_632_se    =res$theta_632_sd[["r2"]],
      r2_632_ci.025=res$theta_632_ci["r2", "0.025"],
      r2_632_ci.25 =res$theta_632_ci["r2", "0.25"],
      r2_632_ci.75 =res$theta_632_ci["r2", "0.75"],
      r2_632_ci.975=res$theta_632_ci["r2", "0.975"],
      
      bic=bic),
    t(data.frame(coef)))
}

get_coef<-function(coefs=NA){
  #     get_coef(coef_glm)
  #     get_coef(coef_enet)
  #     get_coef()
  coefs_all = rep(NA, length(PREDICTORS_FULL))
  names(coefs_all)  = PREDICTORS_FULL
  if(!is.null(coefs)){
    coefs_names = names(coefs)[names(coefs) %in% PREDICTORS_FULL]
    coefs_all[coefs_names] = coefs[coefs_names]
  }
  return(coefs_all)
}

##############################################################################################################
## LOOP
##############################################################################################################

RES = NULL
RES_STRUCT = list()
for(TARGET in TARGETS){
  RES_STRUCT[[TARGET]] = list() 
  #for(PREDICTORS_STR in names(SETTINGS))
  #  RES_STRUCT[[TARGET]][[PREDICTORS_STR]] = list()
  #RES_STRUCT[[TARGET]][["RPT"]] = list()
}

#TARGET = TARGETS[4]
for(PREDICTORS_STR in names(SETTINGS)){
    cat("** PREDICTORS_STR:", PREDICTORS_STR, "**\n" )
  #PREDICTORS_STR = "BASELINE+NIGLOB"
  for(TARGET in TARGETS){
    cat("** TARGET:", TARGET, "**\n" )
    BASELINE = strsplit(TARGET, "[.]")[[1]][1]
    PREDICTORS = unique(c(BASELINE, SETTINGS[[PREDICTORS_STR]]))
    D = db$DB[!is.na(db$DB[, TARGET]),]
    X = as.matrix(D[, PREDICTORS])
    y = D[, TARGET]
    
    ## Scores
    # Bootstrap MSE & R2
    boot_glm = bootpred_ci(x=X,y=y, nboot=NBOOT, mod_fit=fit_glm, mod_predict=predict_glm, theta=losses)
    mod_glm = fit_glm(X, y)
    bic_glm = bic(sum((y - predict_glm(mod_glm, X))^2), n = length(y), p=dim(X)[2])
    coef_glm = get_coef(mod_glm$coefficients)
    if(PREDICTORS_STR == "BASELINE"){
      RES_STRUCT[[TARGET]][[PREDICTORS_STR]] = boot_glm
    }
    
    coef_enet = NULL
    boot_enet = tryCatch({
      mod_enet = fit_enet(X, y)
      bic_enet = bic(sum((y - predict_enet(mod_enet, X))^2), n = length(y), p=dim(X)[2])
      coef_enet = as.double(mod_enet$beta); names(coef_enet) = rownames(mod_enet$beta);# coef_enet = coef_enet[coef_enet!=0]
      coef_enet = get_coef(coef_enet)
      boot_enet  = bootpred_ci(x=X,y=y, nboot=NBOOT, mod_fit=fit_enet, mod_predict=predict_enet, theta=losses)
      RES_STRUCT[[TARGET]][[PREDICTORS_STR]] = boot_enet
      boot_enet
    }, error=function(cond){return(NULL)})
    
    boot_simple =tryCatch({
      mod_simple = fit_simple(X, y)
      bic_simple = bic(sum((y - predict_simple(mod_simple, X))^2), n = length(y), p=3)
      boot_simple = bootpred_ci(x=X,y=y, nboot=NBOOT, mod_fit=fit_simple, mod_predict=predict_simple, theta=losses)
      RES_STRUCT[[TARGET]][["RPT"]] = boot_simple
      boot_simple
    }, error=function(cond)NULL)
    
    #bootpred(x=X,y=y, nboot=100, theta.fit=fit_enet, theta.predict=predict_enet, err.meas=square.err)
    
    RES = rbind(RES,
      format_tuple(TARGET, PREDICTORS_STR, "GLM", boot_glm, bic_glm, coef_glm),
      format_tuple(TARGET, PREDICTORS_STR, "ENET", boot_enet, bic_enet, coef_enet),
      format_tuple(TARGET, PREDICTORS_STR, "RPT", boot_simple, bic_simple, get_coef(NULL)))
  }
}


###########################################################################################################################
## Models comparision
###########################################################################################################################

preffix_name<-function(x, preffix){
  names(x) = c(paste(preffix, names(x), sep="_"))
  return(x)
}

ci_of_diff<-function(x1, x2){
  #H0: theta1 == theta2
  #Ha theta1 != theta2
  ci_=ci(x1 - x2, perc=c(0.025, 0.975))
  r = c(ci_["r2", ], ci_["mse", ])
  names(r) = c(paste("r2", names(r)[1:2], sep="_"), paste("mse", names(r)[3:4], sep="_"))
  return(r)
}

COMPARISONS = NULL
for(TARGET in TARGETS){
  thetas = list()
  thetas[[1]] = RES_STRUCT[[TARGET]][["BASELINE"]]
  thetas[[2]] = RES_STRUCT[[TARGET]][["BASELINE+NIGLOB"]]
  thetas[[3]] = RES_STRUCT[[TARGET]][["BASELINE+NIGLOBFULL"]]
  
  thetas[[4]] = RES_STRUCT[[TARGET]][["BASELINE+CLINIC"]]
  thetas[[5]] = RES_STRUCT[[TARGET]][["BASELINE+CLINIC+NIGLOB"]]
  thetas[[6]] = RES_STRUCT[[TARGET]][["BASELINE+CLINIC+NIGLOBFULL"]]
  
  theta_7 = RES_STRUCT[[TARGET]][["RPT"]]

  v = unlist(lapply(thetas, function(x)x$theta_632_mu["r2"]))
  theta_max = thetas[[which(v==max(v))]]
  as.list(c(preffix_name(thetas[[1]]$theta_632_mu, "1_theta_632_mu"), preffix_name(thetas[[2]]$theta_632_mu, "2_theta_632_mu")))

  COMPARISONS = rbind(COMPARISONS,
  data.frame(TARGET=TARGET, Model1=1, Model2=2, as.list(c(
             ci_of_diff(thetas[[1]]$theta_star, thetas[[2]]$theta_star),
             preffix_name(thetas[[1]]$theta_632_mu, "theta_632_mu_1"), preffix_name(thetas[[2]]$theta_632_mu, "theta_632_mu_2")) )),
  
  data.frame(TARGET=TARGET, Model1=2, Model2=3, as.list(c(
    ci_of_diff(thetas[[2]]$theta_star, thetas[[3]]$theta_star),
    preffix_name(thetas[[2]]$theta_632_mu, "theta_632_mu_1"), preffix_name(thetas[[3]]$theta_632_mu, "theta_632_mu_2")))),
  
  data.frame(TARGET=TARGET, Model1=4, Model2=5, as.list(c(
    ci_of_diff(thetas[[4]]$theta_star, thetas[[5]]$theta_star),
    preffix_name(thetas[[4]]$theta_632_mu, "theta_632_mu_1"), preffix_name(thetas[[5]]$theta_632_mu, "theta_632_mu_2")))),
  
  data.frame(TARGET=TARGET, Model1=5, Model2=6, as.list(c(
    ci_of_diff(thetas[[5]]$theta_star, thetas[[6]]$theta_star),
    preffix_name(thetas[[5]]$theta_632_mu, "theta_632_mu_1"), preffix_name(thetas[[6]]$theta_632_mu, "theta_632_mu_2")))),
  
  data.frame(TARGET=TARGET, Model1="Best linear", Model2="RPT", as.list(c(
    ci_of_diff(theta_7$theta_star, theta_max$theta_star),
    preffix_name(theta_7$theta_632_mu, "theta_632_mu_1"), preffix_name(theta_max$theta_632_mu, "theta_632_mu_2")))) )
}
# Look if zeros lies within the bootraped 95%CI of R2: (r2_0.025, r2_0.975) (I they have the same sign)

##
library(plyr)
RES$target = mapvalues(RES$target, 
                           from = c("TMTB_TIME.CHANGE", "MDRS_TOTAL.CHANGE", "MRS.CHANGE", "MMSE.CHANGE"), 
                           to   = c("TMTB",             "MDRS",              "mRS",        "MMSE"))

RES$model_nb = mapvalues(RES$predictors, 
                          from = c("BASELINE", "BASELINE+NIGLOB", "BASELINE+NIGLOBFULL", "BASELINE+CLINIC", "BASELINE+CLINIC+NIGLOB", "BASELINE+CLINIC+NIGLOBFULL"), 
                          to   = c(1,        2,               3,                   4,               5,                                 6))
RES$model_nb = as.integer(RES$model_nb)

RES[(RES$model == "RPT") & (RES$predictors =="BASELINE+NIGLOB"), "model_nb"] = 7

#write.csv(RES, "/tmp/RES.csv")
## --------------------------------------------------------------------------------------------------
summary = RES[( ((RES$predictors == "BASELINE") & (RES$model =="GLM")) |
            (RES$model =="ENET") | 
            ((RES$predictors == "BASELINE+NIGLOB") & (RES$model =="RPT"))), ]

dim(summary)[1] == 4 * 7

# reorder levels
# levels(summary$target)
summary$target = factor(summary$target, levels=c("MMSE", "MDRS", "TMTB", "mRS") )

summary = summary[order(summary$model_nb), ]
summary = summary[order(summary$target), ]
summary

colnames(summary) = mapvalues(colnames(summary),
                              from = c("TMTB_TIME", "MDRS_TOTAL", "MRS", "MMSE", "AGE_AT_INCLUSION", "SEX",   "EDUCATION", "SYS_BP", "DIA_BP", "SMOKING", "LDL",    "HOMOCYSTEIN", "HBA1C", "CRP17", "ALCOHOL", "LLV", "BPF", "WMHV", "MBcount"),
                              to   = c("TMTB",      "MDRS",       "mRS",  "MMSE",	"Age",	            "Gender",	"Education", "Sys bp",	"Dia bp",	"Smoking",	"LDL", "Homocy",     	"HBA1C",	"CRP",	"Alcohol", "Llv",	"BPF",	"WMHv",	"MBn")
)

wb = loadWorkbook(paste(OUTPUT, "/prediction_bootstrap.xlsx", sep=""), create=TRUE)
createSheet(wb, "All")
writeWorksheet(wb, RES, "All")
createSheet(wb, "Summary")
writeWorksheet(wb, summary, "Summary")
createSheet(wb, "Comparisons")
writeWorksheet(wb, COMPARISONS, "Comparisons")
saveWorkbook(wb)

###########################################################################################################################
## PLOT R2 of all models
###########################################################################################################################

library(ggplot2)
d = summary
junk = d[1:4, ]
junk[,] = NA
junk$target = levels(d$target)
junk$model_nb = 3.5
junk2 = junk
junk2$model_nb = 6.5
junk = rbind(junk2, junk)

d = rbind(d, junk)

d$model_nb = as.factor(d$model_nb)

palette_col = brewer.pal(10, "Paired")[c(1, 5, 3, 2, 6, 4, 10)]

#palette = c(palette[1:3], "white", palette[4:6],  "white", "slategray4")
palette_col = c(palette_col[1:3],  "white", palette_col[4:6],  "white", palette_col[length(palette_col)])

r2boot = ggplot(d, aes(x = model_nb, y = r2, fill=model_nb)) +
  geom_bar(stat = "identity", position="dodge", limits=c(.1, 1)) +
  geom_errorbar(aes(ymin=r2-r2_se, ymax=r2+r2_se), width=.1) +
  #scale_y_continuous(expand=c(.1, 0))+
  facet_wrap(~target) + scale_fill_manual(values=palette_col) + ggtitle("Bootstrap") + theme(legend.position="none")
#  theme(legend.position="bottom", legend.direction="vertical")
x11(); print(r2boot)

# Significance: is zero lies withis the CIs
d$sinificance="*"
prod_cis = d[, "r2_632_ci.025"] * d[, "r2_632_ci.975"]
d$sinificance[prod_cis < 0] = NA
 
r2632 = ggplot(d, aes(x = model_nb, y = r2_632, fill=model_nb)) +
  geom_bar(stat = "identity", position="dodge", limits=c(.1, 1)) +
  geom_errorbar(aes(ymin=r2_632-r2_632_se, ymax=r2_632+r2_632_se), width=.1) +
  geom_text(aes(x=model_nb, y=r2_632+r2_632_se+0.01, label=sinificance))+
  geom_text(aes(x=model_nb, y=r2_632+r2_632_se+0.02, label=sinificance))+
  geom_text(aes(x=model_nb, y=r2_632+r2_632_se+0.03, label=sinificance))+
  #scale_y_continuous(expand=c(.1, 0))+
  facet_wrap(~target, scales="free") + scale_fill_manual(values=palette_col) + ggtitle("Bootstrap") + theme(legend.position="none")
#  theme(legend.position="bottom", legend.direction="vertical")
x11(); print(r2632)

pdf(paste(OUTPUT, "/r2_prediction_bootstrap.pdf", sep=""), width=10, height=7)
print(r2boot)
print(r2632)
dev.off()

svg(paste(OUTPUT, "/r2_prediction_bootstrap.svg", sep=""), width=7, height=7)
#print(r2boot)
print(r2632)
dev.off()

###########################################################################################################################
## PLOT prediction at individual level and CIs of RPT
###########################################################################################################################


# http://stackoverflow.com/questions/14033551/r-plotting-confidence-bands-with-ggplot

PREDICTORS_STR = "BASELINE+NIGLOB"

TARGET = TARGETS[4]
TARGET = TARGETS[1]

data = NULL
for(TARGET in TARGETS){
  cat("** TARGET:", TARGET, "**\n" )
  
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  PREDICTORS = unique(c(BASELINE, SETTINGS[[PREDICTORS_STR]]))
  M36 = gsub("CHANGE", "M36", TARGET)
  
  D = db$DB[!is.na(db$DB[, TARGET]),]
  X = as.matrix(D[, PREDICTORS])
  change_true = D[, TARGET]
  res_best  = bootpred_ci(x=X, y=change_true, nboot=NBOOT, mod_fit=fit_simple, mod_predict=predict_simple, theta=losses)
  changehat_632_mu = apply(res_best$yhat_632, 2, mean, na.rm=TRUE)
  mod_simple = fit_simple(X, change_true)
  change_hat = predict_simple(mod_simple, X)
  group = as.factor(mod_simple$labels)
  baseline = D[, BASELINE]
  m36 = D[, M36]
  fitted = baseline + change_hat
  err_change_632_se = apply(res_best$yhat_632 - change_true, 2, sd, na.rm=TRUE)
  #apply(res_best$yhat_632, 2, sd, na.rm=TRUE)
  # smooth se by group
  for(g in levels(group)){
    #g = "-6.42857142857143"
    #g = "2.28"
    #library(stats)
    m = group == g
    for(xval in unique(baseline[m])){
      m2 = baseline[m] == xval
      err_change_632_se[m][m2] = mean(err_change_632_se[m][m2])
    }
  }
  fitted_lwr = baseline + change_hat - err_change_632_se
  fitted_upr = baseline + change_hat + err_change_632_se
  print(all(baseline + change_true == m36))

  d = data.frame(target=TARGET, baseline=baseline, m36=m36, group=group, fitted=fitted, fitted_lwr=fitted_lwr, fitted_upr=fitted_upr)
  data = rbind(data, d)
}

data$target = mapvalues(data$target, 
                       from = c("TMTB_TIME.CHANGE", "MDRS_TOTAL.CHANGE", "MRS.CHANGE", "MMSE.CHANGE"), 
                       to   = c("TMTB",             "MDRS",              "mRS",        "MMSE"))
# reorder levels
# levels(data$target)
data$target = factor(data$target, levels=c("MMSE", "MDRS", "TMTB", "mRS") )
data$target = mapvalues(data$target, 
                        from = c("TMTB",                   "MDRS",                    "mRS",                    "MMSE"), 
                        to   = c("TMTB (lower is better)", "MDRS (higher is better)", "mRS (lower is better)", "MMSE (higher is better)"))
#


x11()
d = data[data$target=="MDRS_TOTAL.CHANGE",]
d = data[data$target=="TMTB_TIME.CHANGE",]
p1 <- ggplot(data, aes(baseline, m36))+
  geom_point(aes(baseline, m36, color=group), position = "jitter")+
  #geom_point(aes(baseline, m36, color=group))+
  geom_line(aes(baseline, fitted, color=group), size = 1) +
  geom_ribbon(aes(ymin=fitted_lwr, ymax=fitted_upr, color=group), alpha=0.2, linetype=0)+
  geom_abline(linetype="dotted")+
  facet_wrap(~target, scales="free")

p1

svg(paste(OUTPUT, "/r2_baseline-m36_simple_bootstrap-sd.svg", sep=""), width=7, height=7)
#print(r2boot)
print(p1)
dev.off()

pdf(paste(OUTPUT, "/r2_baseline-m36_simple_bootstrap-sd.pdf", sep=""), width=7, height=7)
print(p1)
dev.off()











































###############################################################################################################################""
## OLDIES
bootpred_ci2 <-function(x, y, nboot, theta.fit, theta.predict, err.meas, ...){
  x <- as.matrix(x)
  n <- length(y)
  saveii <- NULL
  fit0 <- theta.fit(x, y)
  yhat0 <- theta.predict(fit0, x)
  err_app <- mean(err.meas(y, yhat0))
  err1 <- matrix(0, nrow = nboot, ncol = n)
  err2 <- rep(0, nboot)
  #
  Yhat = matrix(NA, nrow=nboot, ncol=length(y))
  Errhat = matrix(NA, nrow=nboot, ncol=length(y))
  for (b in 1:nboot) {
    ii <- sample(1:n, replace = TRUE)
    saveii <- cbind(saveii, ii)
    fit <- theta.fit(x[ii, ], y[ii])
    #
    yhat1 <- theta.predict(fit, x[ii, ])
    yhat2 <- theta.predict(fit, x)
    err1[b, ] <- err.meas(y, yhat2)
    err2[b] <- mean(err.meas(y[ii], yhat1))
    #
    Yhat[b, -ii] = theta.predict(fit, x[-ii, , drop=FALSE])
    Errhat[b, -ii] = err.meas(y[-ii], Yhat[b, -ii])
  }
  
  print(max(abs(err1 - Errhat), na.rm=TRUE))
  
  optim <- mean(apply(err1, 1, mean) - err2)
  junk <- function(x, i) {
    sum(x == i)
  }
  # original code average over bootstrap then average over samples
  err_hat_orig = 0
  for (i in 1:n) {
    o <- apply(saveii, 2, junk, i)
    if (sum(o == 0) == 0) 
      cat("increase nboot for computation of the .632 estimator", fill = TRUE)
    err_hat_orig = err_hat_orig + sum(Errhat[o == 0, i])/sum(o == 0)
  }
  err_hat_orig = err_hat_orig / n
  err_632_orig <- 0.368 * err_app + 0.632 * err_hat_orig
  
  mean_err_samples <- function(err, ...){return(mean(err, na.rm=TRUE))}
  
  sd_err_samples <- function(err, coef, n){
    return(sqrt(var(err, na.rm=TRUE)/sum(!is.na(err)) ))
  }
  err_hat_star = apply(Errhat, 1, mean_err_samples)
  err_hat = mean(err_hat_star)
  err_hat_sd = sd(err_hat_star)
  
  cat("err_hat_orig - err_hat:", err_hat_orig - err_hat, "\n")
  
  mean_err_632_samples <- function(err, err_app){
    errhat = mean(err, na.rm=TRUE)
    return( 0.368 * err_app + 0.632 * errhat)
  }
  sd_err_632_samples <- function(err, coef, n){
    return(sqrt(var(err, na.rm=TRUE)/sum(!is.na(err)) ))
  }
  err_632_star = apply(Errhat, 1, mean_err_632_samples, err_app)
  err_632 = mean(err_632_star)
  err_632_sd = sd(err_632_star)
  cat("err_632_orig - err_632:", err_632_orig - err_632, "\n")
  
  results = list(err_app=err_app, err_632_orig=err_632_orig, err_632=err_632, err_hat=err_hat, err_hat_orig=err_hat_orig,
                 err_hat_sd = err_hat_sd, err_632_sd=err_632_sd )
  #return(list(err_app, optim, err.632, call = call))
  
  ## ci ----------------------------------------------------------------
  perc=c(.001,.01,.025,.05,.10,.50,.90,.95,.975,.99,.999)
  citt<-function(thetastar, sdstar){
    thetahat = mean(thetastar)
    sdhat = sd(thetastar)
    tstar <- sort((thetastar - thetahat)/sdstar)[length(thetastar):1]
    ans <- thetahat - sdhat * tstar
    o <- trunc(length(ans) * perc) + 1
    ans1 <- matrix(ans[o], nrow = 1)
    colnames(ans1) = as.character(perc)
    return(ans1)
  }
  
  xstar = Errhat
  ##
  results[["err_hat_ci"]] = citt(
    thetastar = apply(xstar, 1, mean_err_samples),
    sdstar = apply(xstar, 1, sd_err_samples))
  
  ##
  results[["err_632_ci"]] = citt(
    thetastar = apply(xstar, 1, mean_err_632_samples, err_app),
    sdstar = apply(xstar, 1, sd_err_632_samples, err_app))
  
  return(results)
}