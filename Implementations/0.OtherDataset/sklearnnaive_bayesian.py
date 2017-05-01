import os,csv,math
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

#Reading Data
csv_file_object = csv.reader(open('csvdataset.csv', 'rb'))
data=[]
for row in csv_file_object:
	data.append(row)

#Splitting data into training,test,crossvalidation data 
data=np.array(data)
data=data[1::]
x=data[:,0:len(data[0])-1]
y=data[:,len(data[0])-1]
x=x[:,:].astype(np.float64)
x=(x-np.mean(x,axis=0))/np.std(x,axis=0)
y=y[:].astype(np.float64)
trainx=x[0:int(0.7*len(data))]
trainy=y[0:int(0.7*len(data))]
remainx=x[int(0.7*len(data)):len(data)]
remainy=y[int(0.7*len(data)):len(data)]

from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(trainx,trainy)
predicty=clf.predict(remainx)
cnt=0
for i in range(len(predicty)):
	if(predicty[i]==remainy[i]):
		cnt=cnt+1
print (cnt*1.0)/len(predicty)
from sklearn.metrics import classification_report
print classification_report(predicty,remainy)