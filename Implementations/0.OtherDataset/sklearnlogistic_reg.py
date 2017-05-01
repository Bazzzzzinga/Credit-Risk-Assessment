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
