
# coding: utf-8

# In[1]:

import os,csv,math
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans


# In[2]:

csv_file_object = csv.reader(open('csvdataset.csv', 'rb'))
data=[]
for row in csv_file_object:
	data.append(row)


# In[73]:

data=np.array(data)
data=data[2::]
x=data[:,1:24]
y=data[:,24]
x=x[:,:].astype(np.float64)
x=(x-np.mean(x,axis=0))/np.std(x,axis=0)
y=y[:].astype(np.float64)
trainx=x[0:21000]
trainy=y[0:21000]
remainx=x[21000:30000]
remainy=y[21000:30000]


# In[74]:

trainx1=trainx[trainy==1]
trainx0=trainx[trainy==0]


# In[75]:

len1=(len(trainx1)+89)/90
len0=(len(trainx0)+89)/90


# In[76]:

kmeans1=KMeans(n_clusters=len1,max_iter=300).fit(trainx1)
kmeans0=KMeans(n_clusters=len0,max_iter=300).fit(trainx0)


# In[77]:

cluster_centers=np.concatenate((kmeans1.cluster_centers_, kmeans0.cluster_centers_),axis=0)


# In[78]:

def gaussian_kernel(x,y,sigma=5.0):
	exponent=-np.linalg.norm(x-y)**2/(2*(sigma**2))
	return np.exp(exponent)


# In[95]:

newtrainx=np.zeros((len(trainx),len(cluster_centers)+1))
for i in xrange(len(trainx)):
	newtrainx[i][0]=1
	for j in xrange(len(cluster_centers)):
		newtrainx[i][j+1]=gaussian_kernel(trainx[i],cluster_centers[j])
newtrainx=np.concatenate((newtrainx,trainx),axis=1)

newremainx=np.zeros((len(remainx),len(cluster_centers)+1))
for i in xrange(len(remainx)):
	newremainx[i][0]=1
	for j in xrange(len(cluster_centers)):
		newremainx[i][j+1]=gaussian_kernel(remainx[i],cluster_centers[j])
newremainx=np.concatenate((newremainx,remainx),axis=1)


# In[117]:

logit = LogisticRegression(C=0.95)


# In[118]:

logit.fit(newtrainx,trainy)
a=logit.predict(newremainx)


# In[119]:

acc=0
for i in range(len(a)):
	if a[i]==remainy[i]:
		acc=acc+1
print "Classification Report: " + classification_report(remainy,a)
print "Accuracy is",((acc*1.0)/len(a))*100,"%."


# In[ ]:



