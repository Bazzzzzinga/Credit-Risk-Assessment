import os,csv,math
import numpy as np

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
x=x[:,:].astype(np.float64)
x=(x-np.mean(x,axis=0))/np.std(x,axis=0)
y=y[:].astype(np.float64)
#y.resize((30000,1))
trainx=x[0:21000]
trainy=y[0:21000]
remainx=x[21000:30000]
remainy=y[21000:30000]
crossvalidatex=x[21000:25500]
crossvalidatey=y[21000:25500]
testx=x[25500:30000]
testy=y[25500:30000]

#Implementation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
model = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
model.fit(trainx,trainy)
#print "Accuracy over training data is: ",model.score(trainx,trainy)
print "Accuracy over test data is: ",model.score(remainx,remainy)

