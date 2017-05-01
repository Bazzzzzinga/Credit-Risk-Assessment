
# coding: utf-8

# In[25]:

import os,csv,math
import numpy as np
from sklearn.cluster import KMeans
from sklearn import svm
import random
#Reading Data
csv_file_object = csv.reader(open('my_dataset_4.csv', 'rb'))
data=[]
for row in csv_file_object:
	data.append(row)

#Splitting data into training,test,crossvalidation data 
data=np.array(data).astype(np.float64)
x=data[:,0:len(data[0])-1]
y=data[:,len(data[0])-1]
x=(x-np.mean(x,axis=0))/np.std(x,axis=0)
trainx=x[0:int(0.7*len(data))]
trainy=y[0:int(0.7*len(data))]
remainx=x[int(0.7*len(data)):len(data)]
remainy=y[int(0.7*len(data)):len(data)]
#Implementation
trainx1=trainx[trainy==1]
trainx0=trainx[trainy==0]
    
trainx=np.concatenate((trainx1,trainx0),axis=0)
trainy=np.array([1]*len(trainx1)+[0]*len(trainx0))
len1=(len(trainx1)+9)/10
len0=(len(trainx0)+9)/10
kmeans1=KMeans(n_clusters=len1,max_iter=300).fit(trainx1)
kmeans0=KMeans(n_clusters=len0,max_iter=300).fit(trainx0)


# In[26]:

cluster_centers=np.concatenate((kmeans1.cluster_centers_, kmeans0.cluster_centers_),axis=0)


# In[27]:

def gaussian_kernel(x,y,sigma=5.0):
	exponent=-np.linalg.norm(x-y)**2/(2*(sigma**2))
	return np.exp(exponent)


# In[28]:

from sklearn.metrics import classification_report


# In[29]:

newtrainx=np.zeros((len(trainx),len(cluster_centers)+1))
for i in xrange(len(trainx)):
	newtrainx[i][0]=1
	for j in xrange(len(cluster_centers)):
		newtrainx[i][j+1]=gaussian_kernel(trainx[i],cluster_centers[j])
newtrainx=np.concatenate((newtrainx,trainx),axis=1)
#Try changing penalty parameter
clf = svm.SVC(C=100.0)
clf.fit(newtrainx,trainy)

newremainx=np.zeros((len(remainx),len(cluster_centers)+1))
for i in xrange(len(remainx)):
	newremainx[i][0]=1
	for j in xrange(len(cluster_centers)):
		newremainx[i][j+1]=gaussian_kernel(remainx[i],cluster_centers[j])
newremainx=np.concatenate((newremainx,remainx),axis=1)


# In[30]:

ans=clf.predict(newremainx)


# In[31]:

print(classification_report(remainy,np.array(ans)))


# In[32]:

print(sum(ans[i]==remainy[i] for i in range(len(ans)))*1.0/len(ans))


# In[ ]:



