import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

kyphosis=pd.read_csv('kyphosis.csv')
print(kyphosis)

x=kyphosis.drop('Kyphosis',axis=1)
print(x)
y=kyphosis['Kyphosis']
print(y) 

sns.barplot(x='Kyphosis',y='Age',data=kyphosis)
plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)

from sklearn.ensemble import RandomForestClassifier
dtree=RandomForestClassifier()
dtree.fit(x_train,y_train)

rfc_pred=dtree.predict(x_test)
print(rfc_pred)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,rfc_pred))

print(confusion_matrix(y_test,rfc_pred))