import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 


data=pd.read_csv('Classified_data_1.csv',index_col=0)
print(data.head())

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(data.drop('TARGET CLASS',axis=1))
scaler_features=scaler.transform(data.drop('TARGET CLASS',axis=1))

# df=pd.DataFrame(scaler_features,columns=data.columns[:-1])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(scaler_features,data['TARGET CLASS'],test_size=0.30)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train,y_train)

pred=knn.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(pred,y_test))

print(classification_report(pred,y_test))


error_rate=[]

for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i!=y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
