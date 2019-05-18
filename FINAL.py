#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#DATA-PREPROCESSING
frame = pd.read_csv('history.csv',delimiter=';',parse_dates=['date_time'])
frame['minute'] = frame['date_time'].dt.minute
frame['hour'] = frame['date_time'].dt.hour
frame['day_of_week'] = frame['date_time'].dt.weekday_name
frame['month'] = frame['date_time'].dt.month
frame['minute']=frame['minute']+frame['hour']*60
frame.drop('id',inplace=True,axis=1)

X1=frame.ix[frame['day_of_week']=='Monday']
X2=frame.ix[frame['day_of_week']=='Tuesday']
X3=frame.ix[frame['day_of_week']=='Wednesday']
X4=frame.ix[frame['day_of_week']=='Thursday']
X5=frame.ix[frame['day_of_week']=='Friday']
X6=frame.ix[frame['day_of_week']=='Saturday']
X7=frame.ix[frame['day_of_week']=='Sunday']

x1=X1.iloc[:,[0,3]].values
x2=X2.iloc[:,[0,3]].values
x3=X3.iloc[:,[0,3]].values
x4=X4.iloc[:,[0,3]].values
x5=X5.iloc[:,[0,3]].values
x6=X6.iloc[:,[0,3]].values
x7=X7.iloc[:,[0,3]].values

#TEMPERATURE
#AVG TEMP VS TIME OF THE DAY FOR EACH THE 7 DAYS
plt.figure(figsize=(15,10))
byh = X1.loc[:,['temp','minute']].groupby('minute',as_index=False).mean()
plt.scatter(byh['minute'],byh['temp'])
plt.savefig("MON_MINUTE.jpg")

plt.figure(figsize=(15,10))
byh = X2.loc[:,['temp','minute']].groupby('minute',as_index=False).mean()
plt.scatter(byh['minute'],byh['temp'])
plt.savefig("TUES_MINUTE.jpg")

plt.figure(figsize=(15,10))
byh = X3.loc[:,['temp','minute']].groupby('minute',as_index=False).mean()
plt.scatter(byh['minute'],byh['temp'])
plt.savefig("WED_MINUTE.jpg")

plt.figure(figsize=(15,10))
byh = X4.loc[:,['temp','minute']].groupby('minute',as_index=False).mean()
plt.scatter(byh['minute'],byh['temp'])
plt.savefig("THURS_MINUTE.jpg")

plt.figure(figsize=(15,10))
byh = X5.loc[:,['temp','minute']].groupby('minute',as_index=False).mean()
plt.scatter(byh['minute'],byh['temp'])
plt.savefig("FRI_MINUTE.jpg")

plt.figure(figsize=(15,10))
byh = X6.loc[:,['temp','minute']].groupby('minute',as_index=False).mean()
plt.scatter(byh['minute'],byh['temp'])
plt.savefig("SAT_MINUTE.jpg")

plt.figure(figsize=(15,10))
byh = X7.loc[:,['temp','minute']].groupby('minute',as_index=False).mean()
plt.scatter(byh['minute'],byh['temp'])
plt.savefig("SUN_MINUTE.jpg")

#AVG TEMP VS TIME OF THE DAY FOR ENTIRE DATASET
plt.figure(figsize=(15,10))
byh = frame.loc[:,['temp','minute']].groupby('minute').mean()
plt.scatter(np.array(byh.index),byh['temp'])

plt.figure(figsize=(15,10))
byh = frame.loc[:,['temp','hour']].groupby('hour').mean()
plt.scatter(np.array(byh.index),byh['temp'])

plt.figure(figsize=(15,10))
byh = frame.loc[:,['temp','month']].groupby('month').mean()
plt.scatter(np.array(byh.index),byh['temp'])

#APPLYING POLYNOMIAL REGRESSION TO THE DATASET
df1=X1.groupby('minute',as_index=False).mean()
m1=df1.iloc[:,[0]].values
m2=df1.iloc[:,1].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(m1)
poly_reg.fit(X_poly, m2)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, m2)

# Visualising the Polynomial Regression results
plt.figure(figsize=(20,10))
plt.scatter(m1, m2, color = 'red')
plt.plot(m1, lin_reg.predict(poly_reg.fit_transform(m1)), color = 'blue')
plt.title('Temp Vs Minute Analysis')
plt.xlabel('Mins')
plt.ylabel('Temp')
plt.show()

df2=X2.groupby('minute',as_index=False).mean()
t1=df2.iloc[:,[0]].values
t2=df2.iloc[:,1].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(t1)
poly_reg.fit(X_poly, t2)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, t2)

# Visualising the Polynomial Regression results
plt.figure(figsize=(20,10))
plt.scatter(t1, t2, color = 'red')
plt.plot(t1, lin_reg.predict(poly_reg.fit_transform(t1)), color = 'blue')
plt.title('Temp Vs Minute Analysis')
plt.xlabel('Mins')
plt.ylabel('Temp')
plt.show()

df3=X3.groupby('minute',as_index=False).mean()
w1=df3.iloc[:,[0]].values
w2=df3.iloc[:,1].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(w1)
poly_reg.fit(X_poly, w2)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, w2)

# Visualising the Polynomial Regression results
plt.figure(figsize=(20,10))
plt.scatter(w1, w2, color = 'red')
plt.plot(w1, lin_reg.predict(poly_reg.fit_transform(w1)), color = 'blue')
plt.title('Temp Vs Minute Analysis')
plt.xlabel('Mins')
plt.ylabel('Temp')
plt.show()

df4=X4.groupby('minute',as_index=False).mean()
th1=df4.iloc[:,[0]].values
th2=df4.iloc[:,1].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(th1)
poly_reg.fit(X_poly, th2)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, th2)

# Visualising the Polynomial Regression results
plt.figure(figsize=(20,10))
plt.scatter(th1, th2, color = 'red')
plt.plot(th1, lin_reg.predict(poly_reg.fit_transform(th1)), color = 'blue')
plt.title('Temp Vs Minute Analysis')
plt.xlabel('Mins')
plt.ylabel('Temp')
plt.show()

df5=X5.groupby('minute',as_index=False).mean()
f1=df5.iloc[:,[0]].values
f2=df5.iloc[:,1].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(f1)
poly_reg.fit(X_poly, f2)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, f2)

# Visualising the Polynomial Regression results
plt.figure(figsize=(20,10))
plt.scatter(f1, f2, color = 'red')
plt.plot(f1, lin_reg.predict(poly_reg.fit_transform(f1)), color = 'blue')
plt.title('Temp Vs Minute Analysis')
plt.xlabel('Mins')
plt.ylabel('Temp')
plt.show()

df6=X6.groupby('minute',as_index=False).mean()
sa1=df6.iloc[:,[0]].values
sa2=df6.iloc[:,1].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(sa1)
poly_reg.fit(X_poly, sa2)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, sa2)

# Visualising the Polynomial Regression results
plt.figure(figsize=(20,10))
plt.scatter(sa1, sa2, color = 'red')
plt.plot(sa1, lin_reg.predict(poly_reg.fit_transform(sa1)), color = 'blue')
plt.title('Temp Vs Minute Analysis')
plt.xlabel('Mins')
plt.ylabel('Temp')
plt.show()

df7=X7.groupby('minute',as_index=False).mean()
su1=df7.iloc[:,[0]].values
su2=df7.iloc[:,1].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(su1)
poly_reg.fit(X_poly, su2)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, su2)

# Visualising the Polynomial Regression results
plt.figure(figsize=(20,10))
plt.scatter(su1, su2, color = 'red')
plt.plot(su1, lin_reg.predict(poly_reg.fit_transform(su1)), color = 'blue')
plt.title('Temp Vs Minute Analysis')
plt.xlabel('Mins')
plt.ylabel('Temp')
plt.show()

df=frame.groupby('minute',as_index=False).mean()
w1=df.iloc[:,[0]].values
w2=df.iloc[:,1].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(w1)
poly_reg.fit(X_poly, w2)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, w2)

# Visualising the Polynomial Regression results
plt.figure(figsize=(20,10))
plt.scatter(w1, w2, color = 'red')
plt.plot(w1, lin_reg.predict(poly_reg.fit_transform(w1)), color = 'blue')
plt.title('Temp Vs Minute Analysis')
plt.xlabel('Mins')
plt.ylabel('Temp')
plt.show()

DF=frame.groupby('hour',as_index=False).mean()
w1=DF.iloc[:,[0]].values
w2=DF.iloc[:,1].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(w1)
poly_reg.fit(X_poly, w2)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, w2)

# Visualising the Polynomial Regression results
plt.figure(figsize=(20,10))
plt.scatter(w1, w2, color = 'red')
plt.plot(w1, lin_reg.predict(poly_reg.fit_transform(w1)), color = 'blue')
plt.title('Temp Vs Hour Analysis')
plt.xlabel('Hours')
plt.ylabel('Temp')
plt.show()

#HUMIDITY
plt.figure(figsize=(20,10))
byh = X1.loc[:,['temp','minute']].groupby('minute',as_index=False).mean()
plt.scatter(byh['minute'],byh['temp'])

plt.figure(figsize=(20,10))
byh = X2.loc[:,['temp','minute']].groupby('minute',as_index=False).mean()
plt.scatter(byh['minute'],byh['temp'])

plt.figure(figsize=(20,10))
byh = X3.loc[:,['humidity','minute']].groupby('minute').mean()
plt.scatter(np.array(byh.index),byh['humidity'])

plt.figure(figsize=(20,10))
byh = X4.loc[:,['humidity','minute']].groupby('minute').mean()
plt.scatter(np.array(byh.index),byh['humidity'])

plt.figure(figsize=(20,10))
byh = X5.loc[:,['humidity','minute']].groupby('minute').mean()
plt.scatter(np.array(byh.index),byh['humidity'])

plt.figure(figsize=(20,10))
byh = X6.loc[:,['humidity','minute']].groupby('minute').mean()
plt.scatter(np.array(byh.index),byh['humidity'])

plt.figure(figsize=(20,10))
byh = X7.loc[:,['humidity','minute']].groupby('minute').mean()
plt.scatter(np.array(byh.index),byh['humidity'])
