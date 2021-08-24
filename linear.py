from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score,classification_report
# Load dataset.
dftrain = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\programing\\datasets\\train.csv')  # training data
# remove nan values,service period (0,-1,-2)
dftrain = dftrain.dropna()
dftrain = dftrain.reset_index(drop=True)
dftrain = dftrain.loc[dftrain['ServiceSpan'] != 0]
dftrain = dftrain.loc[dftrain['ServiceSpan'] != -1]
dftrain = dftrain.loc[dftrain['ServiceSpan'] != -2]
dftrain = dftrain.reset_index(drop=True)

dftrain['sex'] = dftrain['sex'].map({'Male': 0, 'Female': 1})
dftrain['Aged'] = dftrain['Aged'].map({'No': 0, 'Yes': 1})
dftrain['Married'] = dftrain['Married'].map({'No': 0, 'Yes': 1})
dftrain['TotalDependents'] = dftrain['TotalDependents'].map({'No': 0, 'Yes': 1})
dftrain['MobileService'] = dftrain['MobileService'].map({'No': 0, 'Yes': 1})
dftrain['CyberProtection'] = dftrain['CyberProtection'].map({'No': 0, 'Yes': 1})
dftrain['HardwareSupport'] = dftrain['HardwareSupport'].map({'No': 0, 'Yes': 1})
dftrain['TechnicalAssistance'] = dftrain['TechnicalAssistance'].map({'No': 0, 'Yes': 1})
dftrain['FilmSubscription'] = dftrain['FilmSubscription'].map({'No': 0, 'Yes': 1})
dftrain['CustomerAttrition'] = dftrain['CustomerAttrition'].map({'No': 0, 'Yes': 1})
dftrain['4GService'] = dftrain['4GService'].map({'No': 0, 'Satellite Broadband': 1, 'Wifi Broadband': 2})
dftrain['SettlementProcess'] = dftrain['SettlementProcess'].map({'Check': 0, 'Bank': 1, 'Card': 2, 'Electronic': 3})
dftrain.pop("ID")
x = dftrain.iloc[:, 0:14].values
y = dftrain.iloc[:, 14].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



#model creation
linear = linear_model.LinearRegression()
linear.fit(x_train,y_train)

dftest = dftrain.copy()
dftest.pop("CustomerAttrition")
prediction = linear.predict(dftest)
cutoff = 0.5

y_prediction = np.where(prediction>cutoff,1,0)
y_actual = dftrain["CustomerAttrition"]

conf_matrix = pd.crosstab(y_actual,y_prediction,
                            rownames=["predicted"],
                            colnames=["actual"],
                            margins=True)

print(conf_matrix)  

acc = accuracy_score(y_actual,y_prediction)
print(f'Accuracy:{acc}')