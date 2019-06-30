import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("testset.csv")
dataset['year'] = pd.DatetimeIndex(dataset['datetime_utc']).year
dataset['month'] = pd.DatetimeIndex(dataset['datetime_utc']).month
dataset['day'] = pd.DatetimeIndex(dataset['datetime_utc']).day
dataset['hour'] = pd.DatetimeIndex(dataset['datetime_utc']).hour
 
X = dataset.iloc[:, [20,21,22,23]].values
yreg = dataset.iloc[:, [2,3,4,5,6,7,8,9,11,12,14,15]].values
ylog= dataset.iloc[:, [1]].values

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer=imputer.fit(yreg)
yreg=imputer.transform(yreg)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
ylog[:, 0] = le.fit_transform(ylog[:, 0].astype(str))
'''onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()'''

dfly=pd.DataFrame(ylog)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, yreg, test_size = 0.2, random_state = 0)
ylog_train, ylog_test = train_test_split(ylog, test_size = 0.2, random_state = 0)

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


from sklearn.linear_model import LinearRegression
lr =LinearRegression()
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)



from sklearn.linear_model import LogisticRegression
model= LogisticRegression(random_state=0, multi_class='multinomial',solver='sag')
model.fit(y_train,ylog_train.astype(int))

ypredclass=model.predict(y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(ylog_test.astype(int),ypredclass)

Xaugust=[]
for i in range(0,24):
    Xaugust.append([2019,8,15,i])
    
Yaugpred=lr.predict(Xaugust)
Yaugcond=model.predict(Yaugpred)

dfau=pd.DataFrame(Yaugpred)
dfcon=pd.DataFrame(Yaugcond)
frames=[dfau,dfcon]
result=pd.concat(frames, axis=1)
result=pd.concat([pd.DataFrame(Xaugust),result],axis=1)
result.columns=['year','month','day','Hour','_dewptm','_fog','_hail','_heatindexm','_hum','_pressurem','_rain','_tempm','_thunder','_vism','_wdird','_conds']

print (result)

# Visualising the Training set results dewptm
plt.scatter(X_train[:, [0]], y_train[:,0], color = 'red')
plt.plot(X_train[:, [0]],(lr.predict(X_train))[:,0] , color = 'blue')
plt.title(' dewptm (Training set)')
plt.xlabel('Year')
plt.ylabel('Dew point')
plt.show()

# Visualising the Training set results dewptm
plt.scatter(X_test[:, [0]], y_test[:,0], color = 'red')
plt.plot(X_test[:, [0]],(lr.predict(X_test))[:,0] , color = 'blue')
plt.title(' dewptm (Test set)')
plt.xlabel('Year')
plt.ylabel('Dew point')
plt.show()

# Visualising the Training set results dewptm
plt.scatter(X_train[:, [0]], y_train[:,1], color = 'red')
plt.plot(X_train[:, [0]],(lr.predict(X_train))[:,1] , color = 'blue')
plt.title('Fog  (Training set)')
plt.xlabel('Year')
plt.ylabel('fog')
plt.show()

plt.scatter(X_test[:, [0]], y_test[:,1], color = 'red')
plt.plot(X_test[:, [0]],(lr.predict(X_test))[:,1] , color = 'blue')
plt.title(' fog (Test set)')
plt.xlabel('Year')
plt.ylabel('Fog')
plt.show()

plt.scatter(X_train[:, [0]], y_train[:,2], color = 'red')
plt.plot(X_train[:, [0]],(lr.predict(X_train))[:,2] , color = 'blue')
plt.title('Hail(Training set)')
plt.xlabel('Year')
plt.ylabel('Hail')
plt.show()

plt.scatter(X_test[:, [0]], y_test[:,2], color = 'red')
plt.plot(X_test[:, [0]],(lr.predict(X_test))[:,2] , color = 'blue')
plt.title(' Hail (Test set)')
plt.xlabel('Year')
plt.ylabel('Hail')
plt.show()

plt.scatter(X_train[:, [0]], y_train[:,3], color = 'red')
plt.plot(X_train[:, [0]],(lr.predict(X_train))[:,3] , color = 'blue')
plt.title('Heatindex(Training set)')
plt.xlabel('Year')
plt.ylabel('Heatindex')
plt.show()

plt.scatter(X_test[:, [0]], y_test[:,3], color = 'red')
plt.plot(X_test[:, [0]],(lr.predict(X_test))[:,3] , color = 'blue')
plt.title(' Heatindex (Test set)')
plt.xlabel('Year')
plt.ylabel('Heatindex')
plt.show()

plt.scatter(X_train[:, [0]], y_train[:,4], color = 'red')
plt.plot(X_train[:, [0]],(lr.predict(X_train))[:,4] , color = 'blue')
plt.title('Humi(Training set)')
plt.xlabel('Year')
plt.ylabel('Humi')
plt.show()

plt.scatter(X_test[:, [0]], y_test[:,4], color = 'red')
plt.plot(X_test[:, [0]],(lr.predict(X_test))[:,4] , color = 'blue')
plt.title(' Humi (Test set)')
plt.xlabel('Year')
plt.ylabel('Humi ')
plt.show()

plt.scatter(X_train[:, [0]], y_train[:,5], color = 'red')
plt.plot(X_train[:, [0]],(lr.predict(X_train))[:,5] , color = 'blue')
plt.title('Pressure(Training set)')
plt.xlabel('Year')
plt.ylabel('Pressure')
plt.show()

plt.scatter(X_test[:, [0]], y_test[:,5], color = 'red')
plt.plot(X_test[:, [0]],(lr.predict(X_test))[:,5] , color = 'blue')
plt.title(' Pressure (Test set)')
plt.xlabel('Year')
plt.ylabel('pressure ')
plt.show()

plt.scatter(X_train[:, [0]], y_train[:,6], color = 'red')
plt.plot(X_train[:, [0]],(lr.predict(X_train))[:,6] , color = 'blue')
plt.title('Rain(Training set)')
plt.xlabel('Year')
plt.ylabel('Rain')
plt.show()

plt.scatter(X_test[:, [0]], y_test[:,6], color = 'red')
plt.plot(X_test[:, [0]],(lr.predict(X_test))[:,6] , color = 'blue')
plt.title(' Rain (Test set)')
plt.xlabel('Year')
plt.ylabel('Rain ')
plt.show()

plt.scatter(X_train[:, [0]], y_train[:,7], color = 'red')
plt.plot(X_train[:, [0]],(lr.predict(X_train))[:,7] , color = 'blue')
plt.title('Temp(Training set)')
plt.xlabel('Year')
plt.ylabel('Temp')
plt.show()

plt.scatter(X_test[:, [0]], y_test[:,7], color = 'red')
plt.plot(X_test[:, [0]],(lr.predict(X_test))[:,7] , color = 'blue')
plt.title(' Temp (Test set)')
plt.xlabel('Year')
plt.ylabel('Temp ')
plt.show()

plt.scatter(X_train[:, [0]], y_train[:,8], color = 'red')
plt.plot(X_train[:, [0]],(lr.predict(X_train))[:,8] , color = 'blue')
plt.title('Thunder(Training set)')
plt.xlabel('Year')
plt.ylabel('Thunder')
plt.show()

plt.scatter(X_test[:, [0]], y_test[:,8], color = 'red')
plt.plot(X_test[:, [0]],(lr.predict(X_test))[:,8] , color = 'blue')
plt.title(' Thunder (Test set)')
plt.xlabel('Year')
plt.ylabel('Thunder ')
plt.show()

plt.scatter(X_train[:, [0]], y_train[:,9], color = 'red')
plt.plot(X_train[:, [0]],(lr.predict(X_train))[:,9] , color = 'blue')
plt.title('Vis(Training set)')
plt.xlabel('Year')
plt.ylabel('Vis')
plt.show()

plt.scatter(X_test[:, [0]], y_test[:,9], color = 'red')
plt.plot(X_test[:, [0]],(lr.predict(X_test))[:,9] , color = 'blue')
plt.title(' Vis (Test set)')
plt.xlabel('Year')
plt.ylabel('Vis ')
plt.show()

plt.scatter(X_train[:, [0]], y_train[:,10], color = 'red')
plt.plot(X_train[:, [0]],(lr.predict(X_train))[:,10] , color = 'blue')
plt.title('Wdird(Training set)')
plt.xlabel('Year')
plt.ylabel('Wdird')
plt.show()

plt.scatter(X_test[:, [0]], y_test[:,10], color = 'red')
plt.plot(X_test[:, [0]],(lr.predict(X_test))[:,10] , color = 'blue')
plt.title(' Wdird (Test set)')
plt.xlabel('Year')
plt.ylabel('Wdird ')
plt.show()









    
