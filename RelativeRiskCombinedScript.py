"""
**Author**: David Marsden

**Last Edited**: June 23, 2018

This script:
1) Defines a function for relative risk regression,
2) Calls it on an external data set, and
3) Runs simulations of relative risk regression. \n
Note: the simulations take a long time. Consider specifying a smaller
number of smulations with the nsims argument. 
"""

# Dependencies
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import patsy
from scipy.stats import gaussian_kde

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
    
# Calling function on external data
carrot = pd.read_stata("https://stats.idre.ucla.edu/stat/stata/faq/eyestudy.dta")
carrot.head(6)
carrot.mean(axis = 0)

# Setting reference categories
# for replication to be same as other analysis 
carrot["carrot0ref"] = 0
carrot.loc[carrot["carrot"]==0, ("carrot0ref")] = 1

carrot["gender2ref"] = 0
carrot.loc[carrot["gender"]==1, ("gender2ref")] = 1

RR("lenses ~ carrot0ref + gender2ref + latitude", "id", carrot)
    # Replicates SAS analysis

# Running logit model for comparison
dv, ivs = patsy.dmatrices("lenses ~ carrot0ref + gender2ref + latitude", 
                       carrot, return_type='matrix')
logisticFit = sm.Logit(dv, ivs).fit(disp=0)
print("Carrot OR:", np.exp(logisticFit.params[1]))
print("Gender OR:", np.exp(logisticFit.params[2]))
print("Latitude OR:", np.exp(logisticFit.params[3]))

# Function for simulations
def simulateRR(nsims, smpSize, catPrevalence, 
               baseRisk, RR1, RR2, errorCoefficient):
    """Runs simulations for relative risk regression with one categorical independent
    variable and one continuous independent variable.
    
    **Arguments**: \n
    1) nsims - the number of simulations to run.
    2) smpSize - the samle size for each simulation.
    3) catPrevalence - the proportion of the population where the categorical variable = 1.
    4) baseRisk - the risk when both independent variables and the error term equals zero.
    5) RR1 - the relative risk for the categorical variable.
    6) RR2 - the relative risk for the continuous variable.
    7) errorCoefficient - the relative risk for the error term.
    """    
    
    simulateRR.estimatesRR1 = []
    simulateRR.estimatesRR2 = []
    coverageRR1 = []
    coverageRR2 = []
    estimatesLogOR1 = []
    estimatesLogOR2 = []
    
    np.random.seed()
    
    def generateData():
        # This generates the data for the simulation
        cat = np.random.binomial(n=1, p=catPrevalence, size=smpSize)
        norm = np.random.normal(size=smpSize)
        error = np.random.normal(size=smpSize)
        
        risk = []
        for i in range(smpSize):
            risk_new = baseRisk
            if cat[i]==1:
                risk_new = risk_new*RR1
            risk_new = risk_new*(RR2**norm[i])
            risk_new = risk_new*(errorCoefficient**error[i])
            if risk_new > 1:
                risk_new = 1
            risk.append(risk_new)
        
        y = []
        for i in range(smpSize):
            y_new = np.random.binomial(n=1, p= risk[i], size = 1)
            y_new = int(y_new)
            y.append(y_new)
        
        idx = range(smpSize)
        df = pd.DataFrame({'cat':cat, 'norm':norm, 'outcome':y, 'idx':idx})
        return df
        
    def checkData():
        # This checks the data for complete or quasi-complete separation
        # A known case where generalized linear models can provide terrible results
        # Note: for the simulations I tend to run this would be extremely rare,
        # occuring roughly 1/100,000 times for the model for which it is most likely.
        # However this would be useful for simulations with smaller samples.
        if (
           ((df["cat"]==1)&(df["outcome"]==1)).any() == False or
           ((df["cat"]==1)&(df["outcome"]==0)).any() == False or
           ((df["cat"]==0)&(df["outcome"]==1)).any() == False or
           ((df["cat"]==0)&(df["outcome"]==0)).any() == False
            ):
            return False
        else:
            return True
        
    for n in range(nsims):
        
        checked = False # This variable indicates if the data has been checked.
        
        # This while loop will generate data until it passes the separation check
        while checked == False:
            df = generateData()
            checked = checkData()
            if checked == False:
                print("One sample discarded due to quasi-complete separation concerns.")
        
        # Running relative risk model on generated and checked data
        RR("outcome ~ cat + norm", "idx", df, printOutput = False) 
        
        estRR1 = RR.rrs["RR"][0]
        estRR2 = RR.rrs["RR"][1]
        
        simulateRR.estimatesRR1.append(estRR1)
        simulateRR.estimatesRR2.append(estRR2)
        
        coveredR1 =  bool(RR1<RR.ci["UCL"][0]) & (RR1>RR.ci["LCL"][0])
        coveredR2 =  bool(RR2<RR.ci["UCL"][1]) & (RR2>RR.ci["LCL"][1])
            
        coverageRR1.append(coveredR1)
        coverageRR2.append(coveredR2)
        
        # Running logit model for comparison
        y, X = patsy.dmatrices("outcome ~ cat + norm", df, return_type='matrix')
        logitFit = sm.Logit(y, X).fit(full_output=0, disp=0)
        estLogOR1 = logitFit.params[1]
        estLogOR2 = logitFit.params[2]
        estimatesLogOR1.append(estLogOR1)
        estimatesLogOR2.append(estLogOR2)

    # Describing simulation results
    print("Number of simulations:", nsims)
    print("Sample size:", smpSize)
    print("Risk for Unexposed:", baseRisk)
    print("Prevalence of Categorical Exposure:", catPrevalence)
    print("Relative Risk of Categorical Exposure:", RR1)
    print("Relative Risk of Continuous Covariate:", RR2)
    print("Relative Risk of Error Term:", errorCoefficient)
    print("Expected Number of Exposed Cases:", smpSize*catPrevalence*baseRisk*RR1)
    print("---")
    
    # Evaluating on log scale, since that's the regression coefficient in the model
    logEstRR1 = np.log(simulateRR.estimatesRR1)
    logEstRR2 = np.log(simulateRR.estimatesRR2)
    
    total1 = 0
    for i in range(len(simulateRR.estimatesRR1)):
        total1 = total1 + (logEstRR1[i] - np.log(RR1))**2    
    mseLogRR1 = total1/len(simulateRR.estimatesRR1)
    print("Categorical variable regression coefficient (beta):")
    print("* RMSE:", mseLogRR1**(1/2))
    
    total2 = 0
    for i in range(len(simulateRR.estimatesRR2)):
        total2 = total2 + (logEstRR2[i] - np.log(RR2))**2    
    mseLogRR2 = total2/len(simulateRR.estimatesRR2)
    
    print("* 95% CI coverage probability:", np.mean(coverageRR1))

    biasLogRR1 = np.mean(logEstRR1) - np.log(RR1)
    print("* Bias:", biasLogRR1)
    
    biasLogRR2 = np.mean(logEstRR2) - np.log(RR2)
    
    biasLogOR1 = np.mean(estimatesLogOR1) - np.log(RR1)
    print("* Bias of logistic regression coefficient for RR:", biasLogOR1)
    
    biasLogOR2 = np.mean(estimatesLogOR2) - np.log(RR2)
    
    print("---")
    print("Continuous variable regression coefficient (beta):")
    print("* RMSE:", mseLogRR2**(1/2))
    print("* 95% CI coverage probability:", np.mean(coverageRR2))
    print("* Bias:", biasLogRR2)    
    print("* Bias of logistic regression coefficient for RR:", biasLogOR2)
    print("==============================================================================")
    
# Executing simulations   
simulateRR(nsims = 10000, smpSize = 500, catPrevalence = .25, 
           baseRisk = .25, RR1 = 1.75, RR2 = 1.25, errorCoefficient = 1.75)

simulateRR(10000, 500, .25, .10, 1.75, 1.25, 1.75)
simulateRR(10000, 500, .25, .05, 1.75, 1.25, 1.75)
simulateRR(10000, 1000, .25, .15, 1.75, 1.25, 1.75)
simulateRR(10000, 250, .25, .15, 1.75, 1.25, 1.75)

simulateRR(10000, 500, .25, .15, 1.75, 1.25, 1.75) 
    # I want to plot this one, but put it 2nd in table

# Plotting estimates of beta from last simulation
trueLogRR = np.log(1.75)
LogRR1 = np.log(simulateRR.estimatesRR1)
density = gaussian_kde(LogRR1)
x = np.arange(trueLogRR-.85, trueLogRR+.85, .001)
plt.plot(x, density(x))
plt.axvline(trueLogRR, ls = "dashed")
plt.text(trueLogRR-0.08, .75, 'True beta', rotation=90, fontsize = 12)
plt.title('Density Plot of Coefficient Estimates', fontsize=20)
plt.xlabel('Beta', fontsize=15)
plt.savefig('simulationGraph.pdf', format='pdf', dpi=1000)
plt.show()