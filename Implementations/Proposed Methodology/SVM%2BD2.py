
# coding: utf-8

# In[13]:

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


# In[14]:

from sklearn.metrics import classification_report


# In[15]:

clf = svm.SVC()
clf.fit(trainx,trainy)


# In[16]:

ans=clf.predict(remainx)


# In[17]:

print(classification_report(remainy,np.array(ans)))


# In[18]:

print(sum(ans[i]==remainy[i] for i in range(len(ans)))*1.0/len(ans))


# In[ ]:



