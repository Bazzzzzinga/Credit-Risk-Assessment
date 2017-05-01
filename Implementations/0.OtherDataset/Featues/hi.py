import os,csv,math
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report

def calculateprobability(x,mean,stdev):
	exponent = np.e**(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
#Reading Data
csv_file_object = csv.reader(open('csvdataset.csv', 'rb'))
data=[]
for row in csv_file_object:
	data.append(row)

#Splitting data into training,test,crossvalidation data 
data=np.array(data)
data=data[1::]
print data
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
random.shuffle(data3)
x=data3[:,0:len(data3[0])-1]
y=data3[:,len(data3[0])-1]
# g=0
# o=1
# for i in y:
# 	if i=="0":
# 		g=g+1
# 	else:
# 		o=o+1
# print g,o
# objects = ('0','1')
# y_pos = np.arange(len(objects))
# y=[g,o]
# import matplotlib.pyplot as plt
# plt.bar(y_pos,y,align='center',alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Frequency')
# plt.title('Frequency Histogram')
# plt.show()

x=x[:,:].astype(np.float64)
x=(x-np.mean(x,axis=0))/np.std(x,axis=0)
y=y[:].astype(np.float64)
trainx=x[0:int(0.7*len(data3))]
trainy=y[0:int(0.7*len(data3))]
remainx=x[int(0.7*len(data3)):len(data3)]
remainy=y[int(0.7*len(data3)):len(data3)]

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