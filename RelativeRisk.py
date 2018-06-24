"""
**Author**: David Marsden

**Last Edited**: June 22, 2018

Contains the RR() function for relative risk regression.
"""

# Dependencies
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

# Relative risk function
def RR(formula, idvar, df, printOutput=True):
    """Performs relative risk regression for dichotomous outcomes. 
    Uses a working poisson model and an empirical ("robust") variance estimator.
    
    **Arguments**: \n
    1) formula - a formula expression for the model.
    2) idvar - an identifier for each indepent observation of the data (typically a row).
    3) df - the name of the pandas dataframe.
    4) printOutput - a boolean argument for whether the function should print the output.
    
    **Example Code**: \n
    import pandas as pd \n
    carrot = pd.read_stata("https://stats.idre.ucla.edu/stat/stata/faq/eyestudy.dta") \n
    RR("lenses ~ carrot + gender + latitude", "id", carrot) \n
    # Note: this choice of reference category is different than the IDRE analysis
    
    **References**: \n
    Lumley, T., Kronmal, R., & Ma, S. (2006). 
    Relative risk regression in medical research: 
    models, contrasts, estimators, and algorithms.
    """
    gee = smf.gee(formula, idvar, df, family=sm.families.Poisson())
    results = gee.fit()
    if printOutput:
        print("Relative Risk Regression")
        print("-----------------------------------------------------------------------------------")
        print(results.summary())
    fits = results.fittedvalues
    if printOutput:
        print("Additional diagnostics:")
        print(sum(fits>1), "observations have fitted probabilities greater than one")
        print((sum(fits>1)/len(fits))*100, "% of observations have fitted probabilities greater than one")
        print("==============================================================================")
        print("Relative Risk:")
    RRs = results.params
    RRs = np.exp(RRs)
    RRs = RRs.to_frame()
    RRs = RRs.rename(columns={0:'RR'})
    RR.rrs = RRs.drop(RRs.index[[0]])
    if printOutput:
        print(RR.rrs)
        print("------------------------------------------------------------------------------")
        print("95% Confidence Intervals for Relative Risk:")
    CIs = results.conf_int()
    CIs = CIs.rename(columns={0: 'LCL', 1: 'UCL'})
    CIs = np.exp(CIs)
    RR.ci = CIs.drop(CIs.index[[0]])
    if printOutput:
        print(RR.ci)
        print("==============================================================================")
