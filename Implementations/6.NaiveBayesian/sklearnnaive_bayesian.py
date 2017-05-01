import os,csv,math
import numpy as np
def calculateprobability(x,mean,stdev):
	exponent = np.e**(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
csv_file_object = csv.reader(open('csvdataset.csv', 'rb'))
data=[]
for row in csv_file_object:
	data.append(row)

#Splitting data into training,test,crossvalidation data 
data=np.array(data)
data=data[2::]
x=data[:,1:24]
y=data[:,24]
x=x[:,:].astype(np.float128)
y=y[:].astype(np.float128)

trainx=x[0:21000]
trainy=y[0:21000]
remainx=x[21000:30000]
remainy=y[21000:30000]

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