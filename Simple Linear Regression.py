#!/usr/bin/env python

# # Simple Linear Regression 

# ## Importing Libraries


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data_set=pd.read_csv(r"C:\Users\jeeva\Documents\Machine Learning\simple_linear.csv")

print(data_set)

X=data_set.iloc[:,:-1].values
y=data_set.iloc[:,-1].values
print(X)
print(y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


y_pred = regressor.predict(X_test)


print(y_pred)


plt.scatter(X_train,y_train,color="red")
plt.show()


plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.xlabel("exp")
plt.ylabel("salary")
plt.title("experience vs salary")
plt.show()


plt.scatter(X_test,y_test,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.xlabel("exp")
plt.ylabel("salary")
plt.title("experience vs salary")
plt.show()


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test,y_pred, color = 'blue') # or plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("CTC vs years of experience")
plt.xlabel("Experience in years")
plt.ylabel("CTC")
plt.show()