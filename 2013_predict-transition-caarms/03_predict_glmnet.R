inputfile = "/home/edouard/data/2013_predict-transition-caarms/data/transitionPREP_ARonly_CAARMSonly.csv"

D = read.csv(inputfile)
D$TRANSITION = as.factor(D$TRANSITION)
lr = glm(TRANSITION~., data=D, family=binomial("logit"))

data.frame(lr$coefficients)
#            lr.coefficients
#(Intercept)     -54.1321960
#X.1.1            16.9251449
#X.1.2            -5.6382305
#X.1.3            13.5472380
#X.2.1           -29.3091480
#X.2.2            12.7352477
#X.3.1             7.1011693
#X.3.2            -4.2146462
#X.3.3             7.3163529
#X.4.1           -19.5866028
#X.4.2            -9.3572122
#X.4.3           -20.8122284
#X.5.1            21.2025129
#X.5.2            -8.1467371
#X.5.3           -24.9478757
#X.5.4             0.3265967
#X.6.1            -0.2194619
#X.6.3            26.1976328
#X.6.4           -30.5663754
#X.7.2            23.6730308
#X.7.3            15.2235602
#X.7.4           -15.8141835
#X.7.5            19.7901506
#X.7.6            -1.1613320
#X.7.7            19.7241159
#X.7.8            10.8064980

library(glmnet)
y = D$TRANSITION
X = as.matrix(D[, colnames(D)!="TRANSITION"])

glmnet(X, y, family="binomial", alpha=1)
lasso = glmnet(X, y, family="binomial", alpha=1, lambda=.1)
lasso$beta

s0
X.1.1  1.48572932
X.1.2  .         
X.1.3  .         
X.2.1  .         
X.2.2  .         
X.3.1  .         
X.3.2  .         
X.3.3  0.72816844
X.4.1  .         
X.4.2  .         
X.4.3 -0.91125409
X.5.1  0.09973454
X.5.2  .         
X.5.3  .         
X.5.4  1.19692119
X.6.1  .         
X.6.3  0.30732773
X.6.4  0.84160477
X.7.2  .         
X.7.3  .         
X.7.4 -1.80317795
X.7.5  .         
X.7.6  0.39886684
X.7.7  0.40950036
X.7.8  .         

