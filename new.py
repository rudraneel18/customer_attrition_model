import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import seaborn as sb
import matplotlib.pyplot as plt
from statsmodels.formula.api import logit
from sklearn.metrics import accuracy_score,classification_report

dftrain = pd.read_csv('train.csv')

dftrain = dftrain.dropna()
dftrain = dftrain.reset_index(drop=True)
dftrain = dftrain.loc[dftrain['ServiceSpan'] != 0]
dftrain = dftrain.loc[dftrain['ServiceSpan'] != -1]
dftrain = dftrain.loc[dftrain['ServiceSpan'] != -2]
dftrain = dftrain.reset_index(drop=True)

dftrain['sex'] = dftrain['sex'].map({'Male':0,'Female':1})
dftrain['Aged'] = dftrain['Aged'].map({'No':0,'Yes':1})
dftrain['Married'] = dftrain['Married'].map({'No':0,'Yes':1})
dftrain['TotalDependents'] = dftrain['TotalDependents'].map({'No':0,'Yes':1})
dftrain['MobileService'] = dftrain['MobileService'].map({'No':0,'Yes':1})
dftrain['CyberProtection'] = dftrain['CyberProtection'].map({'No':0,'Yes':1})
dftrain['HardwareSupport'] = dftrain['HardwareSupport'].map({'No':0,'Yes':1})
dftrain['TechnicalAssistance'] = dftrain['TechnicalAssistance'].map({'No':0,'Yes':1})
dftrain['FilmSubscription'] = dftrain['FilmSubscription'].map({'No':0,'Yes':1})
dftrain['CustomerAttrition'] = dftrain['CustomerAttrition'].map({'No':0,'Yes':1})
dftrain['4GService'] = dftrain['4GService'].map({'No':0,'Satellite Broadband':1,'Wifi Broadband':2})
dftrain['SettlementProcess'] = dftrain['SettlementProcess'].map({'Check':0,'Bank':1,'Card':2,'Electronic':3})


data = dftrain.drop('ID', axis=1)
data = dftrain.drop('sex', axis=1)



x_train, x_test = train_test_split(data ,test_size=0.25,random_state=5)
formula = ('CustomerAttrition ~  Aged + Married + TotalDependents + MobileService + CyberProtection + HardwareSupport + FilmSubscription + TechnicalAssistance + Q("4GService")  + ServiceSpan + QuarterlyPayment +GrandPayment ')

model = logit(formula=formula,data = x_train).fit()

prediction = model.predict(x_test)

cutoff = 0.4777

y_prediction = np.where(prediction>cutoff,1,0)
y_actual = x_test['CustomerAttrition']

acc = accuracy_score(y_actual,y_prediction)
print(acc)
        
        


