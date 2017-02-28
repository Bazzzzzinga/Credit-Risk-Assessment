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
finalx=np.ones((30000,25)) #Add a column of all ones for computation of theta0
finalx[:,1:25]=x
x=finalx
#y.resize((30000,1))
trainx=x[0:21000]
trainy=y[0:21000]
remainx=x[21000:30000]
remainy=y[21000:30000]
crossvalidatex=x[21000:25500]
crossvalidatey=y[21000:25500]
testx=x[25500:30000]
testy=y[25500:30000]

#####################Implementation########################

def g(x):
	return 1.0/(1+np.e**(-x))

def forwardpropogation(theta1,theta2,x=remainx):
	tempa2=g(np.dot(x,theta1.transpose()))
	a2=np.ones((len(tempa2),len(tempa2[0])+1))
	a2[:,1:len(a2[0])]=tempa2
	a3=g(np.dot(a2,theta2.transpose()))
	return a3

def backpropagation(lambdaa,theta1,theta2,x=trainx,y=trainy):
	bigdelta1=0*theta1
	bigdelta2=0*theta2
	for ii in range(len(x)):
		a1=x[ii]
		a1=a1.reshape((len(a1),1))
		tempa2=g(np.dot(theta1,a1))
		a2=np.ones((len(x[0]),1))
		a2[1:len(a2),:]=tempa2
		a3=g(np.dot(theta2,a2))
		delta3=a3-y[ii]
		delta2=np.dot(theta2.transpose(),delta3)*(a2*(1-a2))
		bigdelta2=bigdelta2+np.dot(delta3,a2.transpose())
		bigdelta1=bigdelta1+np.dot(delta2[1:len(delta2),:],a1.transpose()) #[1:len(a1),:]
	D1=bigdelta1/(len(x))
	D1[:,1:len(D1[0])]=D1[:,1:len(D1[0])]+lambdaa*theta1[:,1:len(theta1[0])]
	D2=bigdelta2/(len(x))
	D2[:,1:len(D2[0])]=D2[:,1:len(D2[0])]+lambdaa*theta2[:,1:len(theta2[0])]
	return D1,D2

def neuralnetworks(alpha,iterations,lambdaa):
	theta1=np.random.rand(len(x[0])-1,len(x[0]))#RandomInitialisation
	theta2=np.random.rand(1,len(x[0]))
	#print theta1,theta2
	for i in range(1,iterations+1):
		print i
		D1,D2=backpropagation(lambdaa,theta1,theta2)
		theta1=theta1-alpha*D1
		theta2=theta2-alpha*D2
		#print theta1,theta2
	return theta1,theta2


theta1,theta2=neuralnetworks(0.3,10,0.01)
predictremainy=forwardpropogation(theta1,theta2,remainx)
print predictremainy
for i in range(len(predictremainy)):
	#print predictremainy[i]
	if(predictremainy[i]<0.5):
		predictremainy[i]=0
	else:
		predictremainy[i]=1
print classification_report(remainy,predictremainy)
