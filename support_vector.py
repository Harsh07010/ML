import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets._samples_generator import make_blobs

x,y=make_blobs(n_samples=40,centers=2,random_state=20)

clf=svm.SVC(kernel='linear',C=1)
clf.fit(x,y)

plt.scatter(x[:, 0], x[:, 1],c=y,s=30,cmap=plt.cm.Paired)
#plt.show()

#newData=[[3,4],[5,6]]
#print(clf.predict(newData))

ax=plt.gca()
xlim=ax.get_xlim()
ylim=ax.get_ylim()

xx=np.linspace(xlim[0],xlim[1],30)
yy=np.linspace(ylim[0],ylim[1],30)
YY,XX=np.meshgrid(yy,xx)
xy=np.vstack([XX.ravel(),YY.ravel()]).T
Z=clf.decision_function(xy).reshape(XX.shape)

ax.contour(XX,YY,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--', '-', '--'])

ax.scatter(clf.support_vectors_[:, 0],clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none')
plt.show()
