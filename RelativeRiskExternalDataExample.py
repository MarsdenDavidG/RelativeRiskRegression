"""
Author: David Marsden

Last Edited: June 22, 2018

This script performs relative risk regression on an external dataset. It replicates
analysis performed in SAS and illustrates the RR() function. It then compares it
with logistic regression. The RelativeRisk package must already be installed.
"""

import RelativeRisk
import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
    
# Getting external data
carrot = pd.read_stata("https://stats.idre.ucla.edu/stat/stata/faq/eyestudy.dta")

# Checking external data
carrot.head(6)
carrot.mean(axis = 0) # It matches the SAS analysis

# Setting reference categories
# for replication to be same as other analysis 
carrot["carrot0ref"] = 0
carrot.loc[carrot["carrot"]==0, ("carrot0ref")] = 1

carrot["gender2ref"] = 0
carrot.loc[carrot["gender"]==1, ("gender2ref")] = 1

RelativeRisk.RR("lenses ~ carrot0ref + gender2ref + latitude", "id", carrot)
    # Replicates SAS analysis

# Running logit model for comparison
dv, ivs = patsy.dmatrices("lenses ~ carrot0ref + gender2ref + latitude", 
                       carrot, return_type='matrix')
logisticFit = sm.Logit(dv, ivs).fit(disp=0)
print("Carrot OR:", np.exp(logisticFit.params[1]))
print("Gender OR:", np.exp(logisticFit.params[2]))
print("Latitude OR:", np.exp(logisticFit.params[3]))