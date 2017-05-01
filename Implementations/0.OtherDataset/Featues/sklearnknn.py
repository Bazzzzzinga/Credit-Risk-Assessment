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
data1=[]
data0=[]
for i in data:
	if i[-1]=='1':
		data1.append(i)
	else:
		data0.append(i)
import random 
data3=[]
for i in range(2*len(data1)):
	data3.append(data0[random.randint(0,len(data0))])

for i in data1:
	data3.append(i)
data3=np.array(data3)
np.random.shuffle(data3)
x=data3[:,0:len(data3[0])-1]
y=data3[:,len(data3[0])-1]
x=x[:,:].astype(np.float64)
x=(x-np.mean(x,axis=0))/np.std(x,axis=0)
y=y[:].astype(np.float64)
trainx=x[0:int(0.7*len(data3))]
trainy=y[0:int(0.7*len(data3))]
remainx=x[int(0.7*len(data3)):len(data3)]
remainy=y[int(0.7*len(data3)):len(data3)]

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(24,150), random_state=1)
clf.fit(trainx, trainy)
predicty=clf.predict(remainx)
sum=0
for i in range(len(predicty)):
	if(predicty[i]==remainy[i]):
		sum=sum+1
print "Accuracy is"+str((sum*1.0)/len(predicty))
print classification_report(remainy,predicty)