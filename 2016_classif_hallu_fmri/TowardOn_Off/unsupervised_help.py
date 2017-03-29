# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:38:32 2016

@author: ad247405
"""

import csv


error = np.zeros((165))
for i in range(165):
    if t[i]==p[i]:
        error[i]=0
        
    else:
        error[i]=1  
        
    
a=np.array((subject,t,p,error))
a=a.T
   

df = pd.DataFrame(a,columns=['subject', 'true', 'prediction','error'])
df.to_csv('/neurospin/brainomics/2016_classif_hallu_fmri/toward_on/Logistic_L1_L2_TV_with_HC/0.1_0.1_0.1_classification_file.csv')

plt.plot(df['error'],df['score_3rd'],'o')
plt.ylabel('Score on the 3rd component')


BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri"
y=np.load(os.path.join(BASE_PATH,'toward_on','svm','y.npy'))
subject=np.load(os.path.join(BASE_PATH,'toward_on','svm','subject.npy'))


table_patients= np.zeros((165,3))
table_patients[:,0]=subject
table_patients[:,1]=error
table_patients[:,2]=(projections[:,2])
pop = pd.DataFrame(table_patients,columns=["subject","error","PC_3"])

import statsmodels.api as sm
mlm_full = sm.MixedLM.from_formula('error ~ PC_3', data=pop, groups=pop["subject"]).fit()
print(mlm_full.summary())
#####################################################################
