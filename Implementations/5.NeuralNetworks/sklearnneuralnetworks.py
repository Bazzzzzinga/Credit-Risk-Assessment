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
data=data[:,:].astype(np.float64)
x=data[:,0:24]
y=data[:,24]
x=(x-np.mean(x,axis=0))/np.std(x,axis=0)
#finalx=np.ones((30000,25)) #Add a column of all ones for computation of theta0
#finalx[:,1:25]=x
#x=finalx
#y.resize((30000,1))
trainx=x[0:21000]
trainy=y[0:21000]
remainx=x[21000:30000]
remainy=y[21000:30000]
crossvalidatex=x[21000:25500]
crossvalidatey=y[21000:25500]
testx=x[25500:30000]
testy=y[25500:30000]

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(24,), random_state=1)
clf.fit(trainx, trainy)
predicty=clf.predict(remainx)
sum=0
for i in range(len(predicty)):
	if(predicty[i]==remainy[i]):
		sum=sum+1
print "Accuracy is"+str((sum*1.0)/len(predicty))
print classification_report(remainy,predicty)