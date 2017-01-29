import os,csv,math
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report

#Reading Data
csv_file_object = csv.reader(open('csvdataset.csv', 'rb'))
data=[]
for row in csv_file_object:
	data.append(row)

#Splitting data into training,test,crossvalidation data 
data=np.array(data)
data=data[2::]
x=data[:,0:24]
y=data[:,24]
x=x[:,:].astype(np.float128)
x=(x-np.mean(x,axis=0))/np.std(x,axis=0)
y=y[:].astype(np.float128)
finalx=np.ones((30000,25)) #Add a column of all ones for computation of theta0
finalx[:,1:25]=x
x=finalx
trainx=x[0:21000]
trainy=y[0:21000]
remainx=x[21000:30000]
remainy=y[21000:30000]

#Applying Logistic Regression
logit = LogisticRegression(C=1.0)
logit.fit(trainx,trainy)
a=logit.predict(remainx)

#AccuracyCheck
acc=0
for i in range(len(a)):
	if a[i]==remainy[i]:
		acc=acc+1
print "Classification Report: " + classification_report(remainy,a)
print "Accuracy is",((acc*1.0)/len(a))*100,"%."
