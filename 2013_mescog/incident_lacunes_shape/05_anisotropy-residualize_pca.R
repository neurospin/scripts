### compactness ~ inertie_max_norm * grp
moments = read.csv("/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_moments.csv")
pc = read.csv("/home/ed203246/data/mescog/incident_lacunes_shape/pc1-pc2_clusters.csv")
all(moments$lacune_id == pc$lacune_id)
data = data.frame(compactness=moments$compactness, inertie_max_norm=moments$inertie_max_norm, grp=pc$label)
data$grp = as.factor(data$grp)

summary(aov(compactness ~ inertie_max_norm*grp, data=data))

"
                     Df   Sum Sq  Mean Sq F value   Pr(>F)    
inertie_max_norm      1 0.005108 0.005108  19.020 3.43e-05 ***
grp                   3 0.001044 0.000348   1.296   0.2808    
inertie_max_norm:grp  3 0.002807 0.000936   3.484   0.0191 *  
Residuals            90 0.024168 0.000269                     
---
"

### Moment_Invariant ~ inertie_max_norm
data = read.csv("/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_moments.csv")
col_invar = colnames(data)[grep('Moment_Invariant',  colnames(data))]
Y = as.matrix(data[, col_invar])
x = data$inertie_max_norm
#x = data$fa

maov = manova(Y ~ x)

summary(maov)
#          Df  Pillai approx F num Df den Df    Pr(>F)
#x          1 0.95986   169.38     12     85 < 2.2e-16 ***
#Residuals 96

cte = rep(1, dim(Y)[1])
maov_cte = manova(Y ~ cte - 1)

summary(maov_cte)

SSR = sum(maov_cte$residuals ** 2)
SSF = sum(maov$residuals ** 2)

# Explained variance
1 - SSF / SSR
#0.2481053

## PCA on data residualized by anysotropy
mod_resid = prcomp(maov$residuals)
#prcomp(Y)
pc_resid = data.frame(mod_resid$x[, c("PC1", "PC2")])
colnames(pc_resid) = paste(colnames(pc_resid), "resid_anisotropy", sep="_")

pc = read.csv("/home/ed203246/data/mescog/incident_lacunes_shape/pc1-pc2_clusters.csv")

pc = cbind(pc, pc_resid)
write.csv("/home/ed203246/data/mescog/incident_lacunes_shape/pc1-pc2_clusters.csv")

'
mod_resid$rotation[, c("PC1", "PC2")]
                            PC1          PC2
Moment_Invariant_0  -0.06443531 -0.144355614
Moment_Invariant_1  -0.05267510 -0.116097426
Moment_Invariant_2  -0.02160115 -0.246153490
Moment_Invariant_3   0.13855926  0.216682514
Moment_Invariant_4   0.04709903  0.157948391
Moment_Invariant_5   0.63082423 -0.307147499
Moment_Invariant_6  -0.09705455 -0.184546849
Moment_Invariant_7   0.10251212  0.778029143
Moment_Invariant_8  -0.04995228 -0.152727860
Moment_Invariant_9  -0.73016607  0.007878572
Moment_Invariant_10  0.10245249  0.175297017
Moment_Invariant_11 -0.08478250 -0.212952834

mod_resid$sdev
 [1] 0.278478215 0.166132703 0.137023565 0.122888124 0.092615045 0.058570470 0.044994410 0.035890847 0.018636571
[10] 0.008349450 0.005775270 0.001983939


sdev = mod_resid$sdev
var = (sdev ** 2)
var / sum(var)
[1] 0.5009627508 0.1782925698 0.1212867909 0.0975534897 0.0554097337 0.0221605429 0.0130779629 0.0083212851
 [9] 0.0022436495 0.0004503381 0.0002154605 0.0000254261

# on Y
prcomp(Y)$rotation[, c("PC1", "PC2")]
                            PC1         PC2
Moment_Invariant_0  -0.14294875  0.14622265
Moment_Invariant_1  -0.17968673  0.21200114
Moment_Invariant_2   0.34669576 -0.76158041
Moment_Invariant_3   0.18669825 -0.08229445
Moment_Invariant_4   0.07634929 -0.06692794
Moment_Invariant_5   0.52263794  0.40739309
Moment_Invariant_6  -0.13544378  0.07199359
Moment_Invariant_7   0.17125881 -0.18035649
Moment_Invariant_8  -0.08050099  0.06434588
Moment_Invariant_9  -0.64151083 -0.32937370
Moment_Invariant_10  0.16133586 -0.10317077
Moment_Invariant_11 -0.14671097  0.12436119

prcomp(Y)$sdev
 [1] 0.311429403 0.213290510 0.164794879 0.136382795 0.097488396 0.059083246 0.049985930 0.038501760 0.018806984
[10] 0.016619458 0.005857868 0.002004170

sdev = prcomp(Y)$sdev
var = (sdev ** 2)
var / sum(var)
 [1] 4.710849e-01 2.209648e-01 1.319069e-01 9.034397e-02 4.616213e-02 1.695543e-02 1.213600e-02 7.200144e-03
 [9] 1.717981e-03 1.341571e-03 1.666707e-04 1.950964e-05
'
prcomp(Y)$rotation[, c("PC1", "PC2")]
