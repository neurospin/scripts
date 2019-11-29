
library(lme4)

################# FOR BRAIN AGE ANALYSIS ################

data_r_train_ctrl <-read.csv('/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/data_r_train_ctrl.csv')
data_r_test_ctrl <-read.csv('/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/data_r_test_ctrl.csv')
data_r_uhr_test <-read.csv('/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/data_r_uhr_test.csv')

# pour cohortes de contrôles TRAIN et TEST, en contrôlant pour le sexe et le site

residuals_data_r_train_ctrl <- lapply(data_r_train_ctrl[49:140], function(i) residuals(lmer(i ~ sex + (1|site), data=data_r_train_ctrl)))
residuals_data_r_test_ctrl <- lapply(data_r_test_ctrl[49:140], function(i) residuals(lmer(i ~ sex + (1|site), data=data_r_test_ctrl)))

write.csv(residuals_data_r_train_ctrl, file = "/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/residuals_data_r_train_ctrl.csv",row.names=FALSE)
write.csv(residuals_data_r_test_ctrl, file = "/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/residuals_data_r_test_ctrl.csv",row.names=FALSE)

# pour UHR, en contrôlant pour le sexe, le site, la conso d'antipsychotiques, la conso de cannabis, et pour la dépression

residuals_data_r_uhr_test <- lapply(data_r_uhr_test[49:140], function(i) residuals(lmer(i ~ sex + Eq_Chlorpromazine + MADRS + cannabis_last_month + (1|site), data=data_r_uhr_test)))
write.csv(residuals_data_r_uhr_test, file = "/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/residuals_data_r_uhr_test_dep_also.csv",row.names=FALSE)

# !!! pour UHR, en contrôlant pour le sexe, le site, la conso d'antipsychotiques, la conso de cannabis, mais pas pour la dépression

residuals_data_r_uhr_test <- lapply(data_r_uhr_test[49:140], function(i) residuals(lmer(i ~ sex + Eq_Chlorpromazine + cannabis_last_month + (1|site), data=data_r_uhr_test)))
write.csv(residuals_data_r_uhr_test, file = "/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/residuals_data_r_uhr_test_not_adjusted_for_MADRS.csv",row.names=FALSE)

# pour UHR, en contrôlant pour le sexe, le site, la conso de cannabis, et pas pour antipsychotiques et dépression

residuals_data_r_uhr_test <- lapply(data_r_uhr_test[49:140], function(i) residuals(lmer(i ~ sex + cannabis_last_month + (1|site), data=data_r_uhr_test)))
write.csv(residuals_data_r_uhr_test, file = "/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/residuals_data_r_uhr_test_not_adjusted_for_MADRS_and_antipsychotics.csv",row.names=FALSE)

# pour UHR, en contrôlant pour le sexe, le site, la conso de cannabis, la dépression, mais pas pour les antipsychotiques

residuals_data_r_uhr_test <- lapply(data_r_uhr_test[49:140], function(i) residuals(lmer(i ~ sex + MADRS + cannabis_last_month + (1|site), data=data_r_uhr_test)))
write.csv(residuals_data_r_uhr_test, file = "/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/residuals_data_r_uhr_test_not_adjusted_for_antipsychotics.csv",row.names=FALSE)


