import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data=pd.read_csv('50_Startups.csv')

X=data.iloc[:, :-1]
y=data.iloc[:, 4]

# print(X.head())
# print(y.head())

states=pd.get_dummies(X['State'],drop_first=True)

X=X.drop('State',axis=1)

X=pd.concat([X,states],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
linear=LinearRegression()
linear.fit(X_train,y_train)

y_pred=linear.predict(X_test)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)
