from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

x = dftrain.iloc[:, 0:14].values
y = dftrain.iloc[:, 14].values

highAcc = []

for i in range(1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    #Create Gradient Boosting Classifier
    gb = GradientBoostingClassifier(n_estimators=1250,learning_rate=0.03)

    #Train the model using the training sets
    gb.fit(x_train, y_train)

    #Predict the response for test dataset
    y_pred = gb.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    highAcc.append(acc)

    


print("Final:%.5f"%(max(highAcc)*100))
