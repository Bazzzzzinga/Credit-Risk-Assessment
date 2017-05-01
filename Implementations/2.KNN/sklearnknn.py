import os,csv,math
import numpy as np
from sklearn.metrics import classification_report

#Reading Data
csv_file_object = csv.reader(open('csvdataset.csv', 'rb'))
data=[]
for row in csv_file_object:
	data.append(row)

#Splitting data into training,test,crossvalidation data 
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
crossvalidatex=x[21000:25500]
crossvalidatey=y[21000:25500]
testx=x[25500:30000]
testy=y[25500:30000]

#Implementation
from sklearn.neighbors import KNeighborsClassifier
for i in range(1,51):
	neigh = KNeighborsClassifier(n_neighbors=i)
	neigh.fit(trainx, trainy)
	predicty=neigh.predict(remainx)
	sum=0
	for j in range(len(predicty)):
		if(predicty[j]==remainy[j]):
			sum=sum+1
	print i,"Accuracy is"+str((sum*1.0)/len(predicty))
	print classification_report(remainy,predicty)