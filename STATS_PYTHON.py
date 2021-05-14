##---
##  TITLE: "Statistical Methods for Data Mining"
##  AUTHOR: "Rajesh Jakhotia"
##  COMPANY: "K2 ANALYTICS FINISHING SCHOOL PVT LTD"
##  WEBSITE: "www.k2analytics.co.in"
##  EMAIL: ar.jakhotia@k2analytics.co.in
##---

## -------------------------------------------------------------
import pandas as pd
import numpy as np
import os

os.getcwd()
os.chdir("d:/k2analytics/datafile")

inc_exp = pd.read_csv("Inc_Exp_Data.csv")
inc_exp.head(10)


inc_exp.Mthly_HH_Expense.mean()

inc_exp.Mthly_HH_Expense.median()


mth_exp_tmp = pd.crosstab(index=inc_exp["Mthly_HH_Expense"], columns="count")
mth_exp_tmp.reset_index(inplace=True)
mth_exp_tmp[mth_exp_tmp['count'] == inc_exp.Mthly_HH_Expense.value_counts().max()]


## Standard Deviation
pd.DataFrame(inc_exp.iloc[:,0:5].std().to_frame()).T

## Variance
pd.DataFrame(inc_exp.iloc[:,0:5].var().to_frame()).T

summary = inc_exp.describe(include='all')

## Descriptive Statistics
pd.DataFrame(inc_exp['Highest_Qualified_Member'].value_counts().to_frame()).T


##Proportions
freq = pd.DataFrame(inc_exp['Highest_Qualified_Member'].value_counts())
freq.reset_index(inplace=True)
freq.columns = [freq.columns[1], 'count']
freq['prop'] = freq['count'] / sum(freq['count'])
freq

##histogram
inc_exp['Highest_Qualified_Member'].value_counts().plot(kind='bar')

###CrossTable
pd.crosstab(inc_exp.Highest_Qualified_Member, 
            inc_exp.No_of_Earning_Members,margins =True)

def percConvert(ser):
    return round(ser / float(ser[-1]),2)
cr_tb_per = pd.crosstab(inc_exp.Highest_Qualified_Member,
                        inc_exp.No_of_Earning_Members,
                        margins =True).apply(percConvert, axis=1)
cr_tb_per.iloc[0:len(cr_tb_per)-1,0:len(cr_tb_per)-2]

##histogram
inc_exp['No_of_Earning_Members'].value_counts().plot(kind='bar')

## Percentile Distribution
pd.DataFrame(inc_exp.Annual_HH_Income.describe([0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1])).T

def percentile_distribution(df,var):
    per_distr = pd.DataFrame(df[var].describe([0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]))
    per_distr.reset_index(inplace=True)
    per_distr['var'] = per_distr.columns[1]
    per_distr = (per_distr.pivot_table(index='var', columns=['index'])).iloc[:,0:11]
    per_distr.columns = per_distr.columns.droplevel()
    per_distr = per_distr.reindex(columns=['0%','1%','5%','10%','25%','50%','75%','90%','95%','99%','100%'])
    per_distr.reset_index(inplace=True)
    return per_distr
per_df = percentile_distribution(df = inc_exp,var = 'Annual_HH_Income')
per_df

## Box Plot 1
##pip install plotly --ignore-installed nbformat
import plotly.graph_objs as go
import plotly
plotly.__version__

plotly.offline.plot({
    "data": [go.Box(x=inc_exp.Annual_HH_Income)],
    "layout": go.Layout(title="Annual Income Box Plot")
}, auto_open=True)


## Box Plot 2
import seaborn as sns

bplot = sns.boxplot(y='Annual_HH_Income', x='No_of_Earning_Members', 
                 data=inc_exp, 
                 width=0.5,
                 palette="colorblind")


## Box Plot 3
bplot = sns.boxplot(y='Annual_HH_Income', x='Highest_Qualified_Member', 
                 data=inc_exp, 
                 width=0.5,
                 palette="colorblind")


## Proportions
freq = pd.DataFrame(inc_exp['Highest_Qualified_Member'].value_counts())
freq.reset_index(inplace=True)
freq.columns = [freq.columns[1], 'count']
freq['prop'] = freq['count'] / sum(freq['count'])
freq
##histogram
inc_exp['Highest_Qualified_Member'].value_counts().plot(kind='bar')


## Summary
inc_exp.describe()

## Central Limit Theorem

sample_dst = pd.DataFrame()
for i in range(1,1001):
    temp = inc_exp.iloc[np.random.randint(0, len(inc_exp), size=30)]
    temp['sample_no'] = i
    sample_dst = sample_dst.append(temp)
    del temp

sample_dst.head()

sample_mean = sample_dst.groupby('sample_no', as_index=False).agg({
          "Mthly_HH_Income": "mean", "Mthly_HH_Expense": "mean",
          "No_of_Fly_Members": "mean","Emi_or_Rent_Amt": "mean",
          "Annual_HH_Income": "mean"          
          })
sample_mean.columns = ['sample_no', 'Mean_Ann_Inc','Mean_EMI_Rent','Mean_Fly_Mem',
                       'Mean_MHH_Exp','Mean_MHH_Inc']
###Rearrange the columns
sample_mean = sample_mean.reindex(columns=['sample_no','Mean_MHH_Inc','Mean_MHH_Exp',
                                           'Mean_Fly_Mem','Mean_EMI_Rent','Mean_Ann_Inc'])
###Sample Mean
smean = pd.DataFrame(sample_mean.iloc[:,1:6].mean().to_frame())
smean.reset_index(inplace=True)
smean.columns = ['s_vars','smean']
###Population Mean
pmean = pd.DataFrame(inc_exp.iloc[:,0:5].mean().to_frame())
pmean.reset_index(inplace=True)
pmean.columns = ['p_vars','pmean']
###cbind sample_mean and population_mean
spmean = pd.concat([smean.reset_index(drop=True), pmean], axis=1)
### Ratio of sample_mean and population_mean
spmean['ratio'] = spmean.smean / spmean.pmean
spmean


######If first letter of variable is CAPITAL then it does not show in the variable explorer - start
#SMean = pd.DataFrame(sample_mean.iloc[:,1:6].mean().to_frame().T)
#PMean = pd.DataFrame(inc_exp.mean().to_frame().T)
#SPMean = pd.concat([SMean.reset_index(drop=True), PMean], axis=1)
######If first letter of variable is CAPITAL then it does not show in the variable explorer - end

## Standard Deviation
sample_mean.iloc[:,1:6].std()
sample_mean.iloc[:,1:6].var()

inc_exp.iloc[:,0:5].std()
inc_exp.iloc[:,0:5].var()

## Creating a large sample to validate the relationship between 
## Population Standard Deviation and Sample Mean Standard Deviation 

#import random
random = np.random.choice(100000, 5000, replace=True)

rdst = pd.DataFrame([range(1, 5000),random]).transpose()
rdst.columns = ['sr_no', 'random']

random_sample_dst = pd.DataFrame()
N = 500
for i in range(1,1001):
    temp = rdst.iloc[np.random.randint(0, len(rdst), size=N)]
    temp['sample_no'] = i
    random_sample_dst = random_sample_dst.append(temp)
    del temp

random_sample_mean = random_sample_dst.groupby('sample_no', as_index=False).agg({"random": "mean"})
random_sample_mean.columns = ['sample_no','mean_random']

################################
## Varaince
#sample_var = random_sample_mean.iloc[:,1:2].var()
#popln_var = random_sample_dst.iloc[:,1:2].var()
#popln_var_by_N = popln_var / N

###Sample variance
sample_var = pd.DataFrame(random_sample_mean.iloc[:,1:2].var().to_frame())
sample_var.reset_index(inplace=True)
sample_var.columns = ['sample','s_vars']

###Population variance
popln_var = pd.DataFrame(random_sample_dst.iloc[:,1:2].var().to_frame())
popln_var.reset_index(inplace=True)
popln_var.columns = ['popln','p_vars']


###Population variance by N
popln_var_by_N = pd.DataFrame((popln_var.p_vars / N).to_frame())
popln_var_by_N.reset_index(inplace=True)
popln_var_by_N.columns = ['popln_by_n','pvar_by_n']
popln_var_by_N['popln_by_n'] = 'random'

###cbind sample_mean and population_mean
sp_var = pd.concat([sample_var.reset_index(drop=True), popln_var_by_N], axis=1)

### Ratio of sample_mean and population_mean
sp_var['ratio'] = sp_var.s_vars / sp_var.pvar_by_n
sp_var


## Distribution Plot
##histogram
#import matplotlib.pyplot as plt
#plt.hist(inc_exp.Mthly_HH_Expense, bins=10)
#plt.show()
#plt.hist(sample_mean.Mean_MHH_Exp, bins=10)
#plt.show()


#### Other histogram optional
import seaborn as sns

###Hist
sns.distplot(inc_exp.Mthly_HH_Expense,kde=False, bins=10)
sns.distplot(sample_mean.Mean_MHH_Exp,kde=False, bins=10)

########################################################################################
###Density
sns.distplot(inc_exp.Mthly_HH_Expense,hist = False,kde=True, bins=10,
             kde_kws={"color": "g", "alpha":0.3, "linewidth": 4, "shade":True })

sns.distplot(sample_mean.Mean_MHH_Exp,hist = False,kde=True, bins=10,
             kde_kws={"color": "g", "alpha":0.3, "linewidth": 4, "shade":True })

###### hist + dens
sns.distplot(inc_exp.Mthly_HH_Expense,kde=True, bins=10,
             kde_kws={"color": "g", "alpha":0.3, "linewidth": 5, "shade":True }
             )

sns.distplot(sample_mean.Mean_MHH_Exp, kde=True, bins=10,
             kde_kws={"color": "g", "alpha":0.3, "linewidth": 5, "shade":True }
             )


## Validating Empirical Rule between SD and Mean for Normally Distributed Data
sample_mean.head()

inc_Mean = round(sample_mean.Mean_MHH_Inc.mean(),2)
inc_Mean
inc_SD = round(sample_mean.Mean_MHH_Inc.std(),2)
inc_SD


def mean_sd_fun(df, sd, inc_Mean, inc_SD):    
    df = df[(df['Mean_MHH_Inc'] >= inc_Mean - sd * inc_SD) & (df['Mean_MHH_Inc'] <= inc_Mean + sd * inc_SD)] 
    return df
sample_mean_1SD_subset = mean_sd_fun(df = sample_mean, sd = 1, inc_Mean = inc_Mean, inc_SD = inc_SD)

sample_mean_2SD_subset = mean_sd_fun(df = sample_mean, sd = 2, inc_Mean = inc_Mean, inc_SD = inc_SD)

sample_mean_3SD_subset = mean_sd_fun(df = sample_mean, sd = 3, inc_Mean = inc_Mean, inc_SD = inc_SD)

print('Tot_Cnt =', len(sample_mean))
print('SD1_Cnt =', len(sample_mean_1SD_subset))
print('SD2_Cnt =', len(sample_mean_2SD_subset))
print('SD3_Cnt =', len(sample_mean_3SD_subset))


## Correlation
inc_Exp_Corr = inc_exp['Mthly_HH_Income'].corr(inc_exp['Mthly_HH_Expense'])
inc_Exp_Covar = inc_exp['Mthly_HH_Income'].cov(inc_exp['Mthly_HH_Expense'])
inc_SD = inc_exp.Mthly_HH_Income.std()
exp_SD = inc_exp.Mthly_HH_Expense.std()

inc_Exp_Covar / (inc_SD * exp_SD)

inc_exp.corr()



## Hypothesis Testing
from scipy.stats import norm
import scipy.stats as st
from math import *

mu = 70
sigma = 10

def normcdf(x, mu, sigma):
    t = x-mu;
    y = 0.5*erfc(-t/(sigma*sqrt(2.0)));
    return round(1-y,10)
normcdf(x = 100, mu = mu, sigma = sigma)

def normcdf(x, mu, sigma):
    t = x-mu;
    y = 0.5*erfc(-t/(sigma*sqrt(2.0)));
    return round(y,10)
normcdf(x = 80, mu = mu, sigma = sigma)



## Using Z Score
round(1 - st.norm.cdf(3),10)

round(st.norm.cdf(1),7)

round(norm.ppf(0.9, loc=mu, scale=sigma),5)
## Excel Norm.inv
round(norm.ppf(0.1, loc=mu, scale=sigma),5)


## using Z Score for Samy's problem
round(1-norm.cdf(2.5),9)

## t Test
mydata = pd.read_csv('Internet_Mobile_Time.csv')
mydata = mydata.iloc[:,0:1]

xbar = mydata.Minutes.mean()
sd = mydata.Minutes.std()
n = len(mydata.Minutes)

## Null Hypothesis
mu = 144

## Alternate Hypothesis : Mu not equal to 144
## Calculate the t-statistics
tstat = (xbar - mu)/(sd / (n**(1/2.)))
tstat

## Compare with the critical t-value
#Degrees of freedom
df = n - 1

#p-value after comparison with the t
p = 2 * (1 - st.t.cdf(tstat,df=df))
p

## Assuming alpha = 0.05
## Null Hypothesis is Accepted and alternate may be rejected



###Sample plotly
#import plotly
#import plotly.graph_objs as go
#
#plotly.offline.plot({
#    "data": [go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
#    "layout": go.Layout(title="hello world")
#}, auto_open=True)


#plotly.offline.plot({
#    "data": [go.Box(x=np.random.randn(50))],
#    "layout": go.Layout(title="Box Plot")
#}, auto_open=True)

########################################################### t-test
from scipy.stats import ttest_ind
import math
mydata = pd.read_csv('Luggage.csv')


mydata.describe()

## At 0.05 alpha threshold
ttest = ttest_ind(mydata['WingA'], mydata['WingB'], equal_var=True)
ttest


## At 0.01 alpha threshold##############################################################
import numpy as np, scipy.stats as st
ttest = st.ttest_ind(mydata['WingA'], mydata['WingB'], equal_var=True)
ttest

#import statsmodels.stats.api as aa
#cm = sms.CompareMeans(sms.DescrStatsW(X1), sms.DescrStatsW(X2))
#print cm.tconfint_diff(usevar='unequal')

## Assuming Alternative is mean(wingA) > mean(wingB)
tstat1 = st.ttest_ind(mydata.WingA, mydata.WingB)
print("\n The t-statistic is %.3f and the p-value is %.3f." % tstat1)

## Corroborating the Hypothesis with Box Plot
import matplotlib.pyplot as plt
mydata[['WingA','WingB']].plot(kind='box')
#plt.boxplot([mydata.WingA, mydata.WingB], labels = ['WingA', 'WingB'])

###############creating a sample from population
## Hypothesis Testing for Sample taken from Population
## Let us import the dataset and draw a sample
import random

df = pd.read_csv("hypothesis_test.csv")
popln = df[["Age", "Balance", "No_OF_CR_TXNS", "SCR"]]
popln['random'] = np.random.random(len(popln))
sample_dst = popln[popln['random'] <= 0.1]

len(popln)
len(sample_dst)

from scipy import stats
from scipy.stats import ttest_rel

one_sample = stats.ttest_1samp(sample_dst.Age, popln.Age.mean())
print("\n The t-statistic is %.3f and the p-value is %.3f." % one_sample)

print("\n Sample Mean :", round(sample_dst.Age.mean(),4),",",
      " Population Mean :",round(popln.Age.mean(),4))

one_sample = stats.ttest_1samp(sample_dst.Balance, popln.Balance.mean())
print("\n The t-statistic is %.3f and the p-value is %.3f." % one_sample)

print("\n Sample Mean :", round(sample_dst.Balance.mean(),4),",",
      " Population Mean :",round(popln.Balance.mean(),4))


one_sample = stats.ttest_1samp(sample_dst.No_OF_CR_TXNS, popln.No_OF_CR_TXNS.mean())
print("The t-statistic is %.3f and the p-value is %.3f." % one_sample)

print("\n Sample Mean :", round(sample_dst.No_OF_CR_TXNS.mean(),4),",",
      " Population Mean :",round(popln.No_OF_CR_TXNS.mean(),4))


one_sample = stats.ttest_1samp(sample_dst.SCR, popln.SCR.mean())
print("The t-statistic is %.3f and the p-value is %.3f." % one_sample)

print("\n Sample Mean :", round(sample_dst.SCR.mean(),4),",",
      " Population Mean :",round(popln.SCR.mean(),4))


## Paired t Test
concrete_dst = pd.read_csv("concrete.csv")

concrete_dst[['SevenDays','TwoDays']].describe()
#concrete_dst[['SevenDays','TwoDays']].plot(kind='box')

two_sample = stats.ttest_rel(concrete_dst.SevenDays, concrete_dst.TwoDays)
print("\n The t-statistic is %.3f and the p-value is %.3f." % two_sample)

round(concrete_dst['SevenDays'].mean(),3)
round(concrete_dst['TwoDays'].mean(),4)


concrete_dst['diff'] = concrete_dst['SevenDays'] - concrete_dst['TwoDays']
concrete_dst['diff'].plot(kind='hist', title= 'Difference Histogram')

one_sample = stats.ttest_1samp(concrete_dst['diff'], 0)
print("The t-statistic is %.3f and the p-value is %.3f." % one_sample)

#stats.probplot(concrete_dst['diff'], plot= plt)
#plt.title('Difference Q-Q Plot')


## Chi-Sq Test
favour = [108, 18, 35, 24]
neutral = [46, 12, 14, 7]
oppose = [71,  30, 26, 9]


sm_wt = pd.DataFrame(
    {'favour': favour,
     'neutral': neutral,
     'oppose': oppose
    })

sm_wt.index = ["Hourly Worker", "Supervisor", 
               "Middle Management", "Upper Managment"]

#save(sm_wt, file="Self_Managed_Work_Teams.RData")

import pickle
#### Save the data
sm_wt.to_pickle("Self_Managed_Work_Teams.pkl")

#### Read the data
sm_wt = pd.read_pickle("Self_Managed_Work_Teams.pkl")
sm_wt

from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(sm_wt)
print("\n\t Pearson's Chi-Squared test \n",
      "Chi2 :",round(chi2,4),",","dof :",dof,",",
      " P Value :",round(p,4))

expected = pd.DataFrame(expected, columns = ['favour','neutral','oppose'],
                  index = ["Hourly Worker", "Supervisor", 
                           "Middle Management", "Upper Managment"])
expected

## Classroom Exercise for Chi-Sq
crs_tb = pd.crosstab(df.Occupation, df.Target)

chi2, p, dof, expected = chi2_contingency(crs_tb, correction=False)
print("\n\t Pearson's Chi-Squared test \n",
      "Chi2 :",round(chi2,4),",","dof :",dof,",",
      " P Value :",round(p,4))

             
