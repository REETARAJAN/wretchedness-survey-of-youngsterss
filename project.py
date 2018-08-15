

import pandas as pd
import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

pd.set_option('display.max_columns',150)
plt.style.use('bmh')
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")
df1 = pd.read_csv('/home/psyec/Downloads/project/responses.csv')
responses_pre = pd.read_csv('/home/psyec/Downloads/project/responses.csv')
columns = pd.read_csv('/home/psyec/Downloads/project/columns.csv')
responses=responses_pre.dropna()
responses.describe()
# Query 1 -------- loneliness by gender -------------
male = responses[responses.Gender == 'male'] 
male
female = responses[responses.Gender == 'female']
female
from scipy.stats import t, norm
def CI(x):
    mean = x.mean()
    std = x.std(ddof = 1)
    n = len(x)
    zstar = norm.ppf(0.975)
    CI_Lower = mean - zstar*std/(n**(1/2))
    CI_Upper = mean + zstar*std/(n**(1/2))
    
    return [CI_Lower, CI_Upper]
    
## build a function to calculate 95% confidence interval for difference between two groups   
def twosampCI(x,y):
   xdiff = x.mean() - y.mean()
   x_s = x.std(ddof=1)
   y_s = y.std(ddof=1)
   std_error = (((x_s**2)/len(x)) + ((y_s**2)/len(y)))**0.5
   zstar = norm.ppf(0.975)
   CI_Lower = xdiff - zstar*std_error
   CI_Upper = xdiff + zstar*std_error
   
   return [CI_Lower, CI_Upper]
CI(male.Loneliness)
CI(female.Loneliness)
twosampCI(male.Loneliness, female.Loneliness)

from scipy import stats 
#-----------------------hypothesis testing-----------------------------
def check_values_in_variable(variable):
    uniquevalues= responses[variable].unique()
    return uniquevalues
#Check the unique values in a variable, return the values as a list

def divide_groups(groupingvar):
    uniquevalues = check_values_in_variable(groupingvar)
    groups=[]
    for value in uniquevalues:
        groups.append(responses.groupby([groupingvar]).get_group(value))
    return groups
#Divide the dataset by the grouping variable, using the unique values list
#returned by the above function       
        
def hypothesis_test(groupingvar,interestvar):
    uniquevalues = check_values_in_variable(groupingvar) 
    groups=[]
    groupnames=[]
    varlists=[]
    results=[]
    for value in uniquevalues
        groups.append(responses.groupby([groupingvar]).get_group(value))
        groupnames.append(value)
    for group in groups
        varlists.append(group[interestvar])
    for var1 in varlists 
        for var2 in varlists
            if (var1 is var2)
                continue
            else:
                results.append(stats.ttest_ind(var1,var2))
    for i in range(len(groupnames)): #display the p-values
        for j in range(len(groupnames)):
                if groupnames[i] is groupnames[j]:
                    continue
                else:
                    print('Compare '+ str(interestvar) + ' of ' + str(groupnames[i])+' and '+str(groupnames[j])+":\n")       
                    print(str(results[i]) + ":\n")
    return results[0] #For testing
        
hypothesis_test('Only child','Loneliness')
hypothesis_test('Gender','Loneliness')
col = 'Loneliness'
corr = responses.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
var_of_interest = 'Loneliness'
fig, ax = plt.subplots(figsize=(5,5))
sns.countplot(responses[var_of_interest], orient = 'h')
_ = plt.xticks(fontsize=14)
_ = plt.yticks(fontsize=14)

#%% query 4
from sklearn.cross_validation import KFold, train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

mov_mus   = df1.iloc[:,[0,19]]   
scared    = df1.iloc[:,63:73]   
interests = df1.iloc[:,31:63]    
demo      = df1.iloc[:,140:150]  
spending  = df1.iloc[:,133:140]  

print(responses.columns.get_loc('Loneliness')) 
predict   = df1.iloc[:,99]       
df2 = mov_mus.join([scared, interests, demo, spending, predict])
gender  = pd.get_dummies(df2['Gender'])
handed  = pd.get_dummies(df2['Left - right handed'])
vil_tow = pd.get_dummies(df2['Village - town'])
resid   = pd.get_dummies(df2['House - block of flats'])
educa   = pd.get_dummies(df2['Education'])
df2.drop(['Gender','Left - right handed','Village - town','House - block of flats','Education'], axis=1, inplace=True)
df2 = df2.join([gender, handed, vil_tow, resid, educa])
df2=df2.dropna()
df2.loc[df2['Loneliness'] <= 3, 'Loneliness'] = 0
df2.loc[df2['Loneliness'] > 3, 'Loneliness'] = 1
x = df2.drop('Loneliness', axis=1)
y = df2['Loneliness']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)
kf = KFold(len(x_train), n_folds=5)
logreg = LogisticRegression()
param_grid = {'C':[.01,.03,.1,.3,1,3,10]}
gs_logreg = GridSearchCV(logreg, param_grid=param_grid, cv=kf)
gs_logreg.fit(x_train, y_train)
gs_logreg.best_params_ 
logreg = LogisticRegression(C=.01)
logreg.fit(x_train, y_train)
print('Average accuracy score on cv (KFold) set: {:.3f}'.format(np.mean(cross_val_score(logreg, x_train, y_train, cv=kf))))
print('Accuracy score on test set is: {:.3f}'.format(logreg.score(x_test, y_test)))
coeff_df = pd.DataFrame(data=logreg.coef_[0], index=[x_train.columns], columns=['Feature_Import'])
coeff_df = coeff_df.sort_values(by='Feature_Import', ascending=False)
fig, ax1 = plt.subplots(1,1, figsize=(7,6)) 
sns.barplot(x=coeff_df.index, y=coeff_df['Feature_Import'], ax=ax1)
ax1.set_title('All Features')
ax1.set_xticklabels(labels=coeff_df.index, size=6, rotation=90)
ax1.set_ylabel('Importance') 
coeff_df['Feature_Import'].tail(10)         
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,10))
sns.barplot(x=coeff_df.index[:10], y=coeff_df['Feature_Import'].head(10), ax=ax1)
ax1.set_title('Top Positive Features')
ax1.set_ylabel('Importance')
ax1.set_xticklabels(labels=coeff_df.index[:10], fontsize=8, rotation=20)
sns.barplot(x=coeff_df.index[-10:], y=coeff_df['Feature_Import'].tail(10), ax=ax2, palette='hls')
ax2.set_title('Top Negative Features')
ax2.set_ylabel('Importance')
ax2.set_xticklabels(labels=coeff_df.index[-10:], fontsize=8, rotation=20)
def areyoulonely():
    print(' ====== LONELINESS TEST  =======') 
    print('Do you think you are lonely?')
    temp=input() 
    print('')
    print('Doesn\'t matter what you think.')
    print('We will see after 8 questions.')
     #store user input into each variable:
    x1_public_speaking = int(input('1. Public speaking: Not afraid at all 1-2-3-4-5 Very afraid of (integer)'))
    x2_writing=int(input('2. Poetry writing: Not interested 1-2-3-4-5 Very interested (integer)'))
    x3_internet=int(input('3. Internet: Not interested 1-2-3-4-5 Very interested (integer)'))
    x4_PC=int(input('4. PC Software, Hardware: Not interested 1-2-3-4-5 Very interested (integer)'))
    x5_fun_fri=int(input('5. Socializing: Not interested 1-2-3-4-5 Very interested (integer)'))
    x6_eco_man=int(input('6. Economy, Management: Not interested 1-2-3-4-5 Very interested (integer)'))
    x7_cars=int(input('7. Cars: Not interested 1-2-3-4-5 Very interested (integer)'))
    x8_ent_sp=int(input('8. I spend a lot of money on partying and socializing.: Strongly disagree 1-2-3-4-5 Strongly agree (integer)'))    
calculateloneliness(x1_public_speaking,x2_writing,x3_internet,x4_PC,x5_fun_fri,x6_eco_man,x7_cars,x8_ent_sp)
    return x1_public_speaking,x2_writing,x3_internet,x4_PC,x5_fun_fri,x6_eco_man,x7_cars,x8_ent_sp

def calculateloneliness(x1,x2,x3,x4,x5,x6,x7,x8):

    yscore=0.00055438+0.133*x1+0.108*x2+0.0924*x3+0.0919*x4-0.102*x5-0.102*x6-0.11*x7-0.128*x8
    print()
    print(' ==========  RESULT  =========')
    if yscore>0.5
        print('Hi Mr./Ms. Lonely !')

        print('You are '+str('{percent:.2%}'.format(percent=yscore)+' likely to be a lonely person.') )
    else
        print('You must be a joyful person!')

        print('You are only'+str('{percent:.2%}'.format(percent=yscore)+' likely to be a lonely person.'))
    outputfortest=int(yscore)
    return outputfortest

areyoulonely()
 
        


