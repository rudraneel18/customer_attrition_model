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

a=[]
data = dftrain.drop('ID', axis=1)
data = dftrain.drop('GrandPayment', axis=1)
data = dftrain.drop('SettlementProcess', axis=1)
acch=0.83
for i in range(100):
    for j in range(10,25):
        
        x_train, x_test = train_test_split(data ,test_size=j/50, random_state=i)

        formula = ('CustomerAttrition ~ sex+ Aged + Married + TotalDependents + MobileService + CyberProtection + HardwareSupport + FilmSubscription + TechnicalAssistance + Q("4GService") + ServiceSpan + QuarterlyPayment')


        model = logit(formula=formula,data = x_train).fit()

        prediction = model.predict(x_test)

        cutoff = 0.5


        y_prediction = np.where(prediction>cutoff,1,0)
        y_actual = x_test['CustomerAttrition']
        acc = accuracy_score(y_actual,y_prediction)
        if (acc>acch):
            a.append([i,j])
for i in range(len(a)):
    print(a[i])
