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
data=data[1::]
x=data[:,0:len(data[0])-1]
y=data[:,len(data[0])-1]
x=x[:,:].astype(np.float64)
x=(x-np.mean(x,axis=0))/np.std(x,axis=0)
y=y[:].astype(np.float64)
y=y.reshape(y.shape[0],1)
finalx=np.ones((len(data),len(data[0]))) #Add a column of all ones for computation of theta0
finalx[:,1:len(data[0])]=x
x=finalx
trainx=x[0:int(0.7*len(data))]
trainy=y[0:int(0.7*len(data))]
remainx=x[int(0.7*len(data)):len(data)]
remainy=y[int(0.7*len(data)):len(data)]

#Computing hthetaofx for logistic regression where htheta(x)=1/(1+e**(-thetatranspose*x))
def hthetaofx(x,theta):
	return 1.0/(1+np.e**(-1*(np.dot(x,theta))))

#Computing Costerrorfunction
def joftheta(x,theta,y):
	k=hthetaofx(x,theta)
	step1=y*np.log(k)
	step2=(np.ones((len(y),1))-y)*(np.log(np.ones((len(y),1))-k))
	return (-1.0/len(y))*((step1+step2).sum())

#Applying Logistic Regression
def logistic_regression(alpha,iterations,x=trainx,y=trainy):
        theta=np.zeros((len(x[0]),1))
        while iterations:
                iterations=iterations-1
                theta=theta-alpha*(np.dot(x.transpose(),(1.0/(1+np.e**(-1*(np.dot(x,theta))))-y)))
                #print joftheta(x,theta,y)
        return theta
theta=logistic_regression(0.00001,1000)

#AccuracyCheck
a=np.dot(remainx,theta)
acc=0
for i in range(len(a)):
	if a[i]<0:
		a[i]=0
	else:
		a[i]=1
	if a[i]==remainy[i]:
		acc=acc+1
print "Accuracy is",((acc*1.0)/len(a))*100,"%."
print classification_report(remainy,a)
