
# coding: utf-8

# In[1]:

import os,csv,math


# In[2]:

import numpy as np
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.metrics import classification_report


# In[3]:

csv_file_object = csv.reader(open('csvdataset.csv', 'rb'))


# In[4]:

data=[]


# In[5]:

for row in csv_file_object:
	data.append(row)


# In[6]:

data=np.array(data)


# In[7]:

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


# In[8]:

trainx1=trainx[trainy==1]
trainx0=trainx[trainy==0]


# In[9]:

len1=(len(trainx1)+99)/100
len0=(len(trainx0)+99)/100


# In[10]:

kmeans1=KMeans(n_clusters=len1,max_iter=300).fit(trainx1)


# In[11]:

kmeans0=KMeans(n_clusters=len0,max_iter=300).fit(trainx0)


# In[12]:

cluster_centers=np.concatenate((kmeans1.cluster_centers_, kmeans0.cluster_centers_),axis=0)


# In[15]:

def gaussian_kernel(x,y,sigma=5.0):
	exponent=-np.linalg.norm(x-y)**2/(2*(sigma**2))
	return np.exp(exponent)


# In[16]:

newtrainx=np.zeros((len(trainx),len(cluster_centers)+1))


# In[17]:

for i in xrange(len(trainx)):
	newtrainx[i][0]=1
	for j in xrange(len(cluster_centers)):
		newtrainx[i][j+1]=gaussian_kernel(trainx[i],cluster_centers[j])


# In[22]:

newtrainx=np.concatenate((newtrainx,trainx),axis=1)


# In[23]:

clf = svm.SVC(C=1)


# In[24]:

clf.fit(trainx,trainy)


# In[25]:

ans=clf.predict(remainx)
print(classification_report(remainy,np.array(ans)))
print(sum(ans[i]==remainy[i] for i in range(len(ans)))*1.0/len(ans))


# In[ ]:



