from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

import pandas as pd

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


dftrain.pop('ID')
data = dftrain.drop('GrandPayment', axis=1)
data = dftrain.drop('SettlementProcess', axis=1)
data = dftrain.drop('Married', axis=1)

y=dftrain.pop('CustomerAttrition')
x = dftrain

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(x_train, y_train)
data = tree.export_graphviz(dtree, out_file=None, feature_names=dftrain.columns)


prediction = dtree.predict(x_test)
actual = y_test
conf_matrix = pd.crosstab(actual,prediction,rownames=["predicted"],colnames=["actual"],margins=True)
print(conf_matrix)  

acc = accuracy_score(actual,prediction)
print(f'Accuracy:{acc}')