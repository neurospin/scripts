86 CTL et 108 patient



/neurospin/brainomics/2014_bd_dwi/enettv_bd_dwi_site
====================================================

Le site en covar
k = 10000, a=0.1, l1=0.04, l2=0.36, tv=0.6
------------------------------------------
tx0 = 0.61; tx1=0.66

succ0 = as.integer(tx0 * 86)
succ1 = as.integer(tx1 * 108)
binom.test(succ0, 86, p = 0.5, alternative ="greater")
binom.test(succ1, 108, p = 0.5, alternative ="greater")
p-value = 0.03
p-value = 0.0006908

k = -1, a=0.05, l1=0.2, l2=0.2, tv=0.6
------------------------------------------
tx0 = 0.55; tx1=0.69

succ0 = as.integer(tx0 * 86)
succ1 = as.integer(tx1 * 108)
binom.test(succ0, 86, p = 0.5, alternative ="greater")
binom.test(succ1, 108, p = 0.5, alternative ="greater")
p-value = 0.2253
p-value = 7.455e-05

=> filtrage univarié

/neurospin/brainomics/2014_bd_dwi/enettv_bd_dwi_trunc
======================================================
Le site en covar

k = -1, a=0.05, l1=0.2, l2=0.2, tv=0.6
------------------------------------------
tx0 = 0.59; tx1=0.70
succ0 = as.integer(tx0 * 86)
succ1 = as.integer(tx1 * 108)
binom.test(succ0, 86, p = 0.5, alternative ="greater")
binom.test(succ1, 108, p = 0.5, alternative ="greater")

p-value = 0.08029
p-value = 3.275e-05

*********************************************************************
k = -1, a=1, l1=0.01, l2=0.09, tv=0.9
------------------------------------------
tx0 = 0.64; tx1=0.63
succ0 = as.integer(tx0 * 86)
succ1 = as.integer(tx1 * 108)
binom.test(succ0, 86, p = 0.5, alternative ="greater")
binom.test(succ1, 108, p = 0.5, alternative ="greater")
p-value = 0.006336
p-value = 0.004529
*********************************************************************

k = 100000, a=0.1, l1=0.0095, l2=0.9405, tv=0.05
------------------------------------------
tx0 = 0.64; tx1=0.62
succ0 = as.integer(tx0 * 86)
succ1 = as.integer(tx1 * 108)
binom.test(succ0, 86, p = 0.5, alternative ="greater")
binom.test(succ1, 108, p = 0.5, alternative ="greater")
p-value = 0.006336
p-value = 0.01321


/neurospin/brainomics/2014_bd_dwi/enettv_bd_dwi_skel
====================================================
k = -1, a=0.5, l1=0.0, l2=0.95, tv=0.05
------------------------------------------
tx0 = 0.67; tx1=0.61
succ0 = as.integer(tx0 * 86)
succ1 = as.integer(tx1 * 108)
binom.test(succ0, 86, p = 0.5, alternative ="greater")
binom.test(succ1, 108, p = 0.5, alternative ="greater")
p-value = 0.001677
p-value = 0.0214



TODO
====
Reprendre k = -1, a=1, l1=0.01, l2=0.09, tv=0.9
- génèrer les cartes de poids
- Tableau sujet, CTL/delirant/non délirant vs proba post
=> PF (Psychotic Features) dans DB_clinic.xls
Essayer de classer les BP qui délirent de ceux qui ne délirent pas



