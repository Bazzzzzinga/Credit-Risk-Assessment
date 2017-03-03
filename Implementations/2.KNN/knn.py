import os,csv,math
import numpy as np

#Reading Data
csv_file_object = csv.reader(open('csvdataset.csv', 'rb'))
data=[]
for row in csv_file_object:
	data.append(row)

data=np.array(data)
data=data[2::]
x=data[:,1:24]
y=data[:,24:25]
x=x[:,:].astype(np.float64)
x=(x-np.mean(x,axis=0))/np.std(x,axis=0)
y=y[:].astype(np.float64)
trainx=x[0:21000]
trainy=y[0:21000]
remainx=x[21000:30000]
remainy=y[21000:30000]

distance0=[]
distance1=[]
for j in range(len(remainx)):
	d0=[]
	d1=[]
	print j
	for i in range(len(trainx)):
		temp=(np.square(trainx[i]-remainx[j])).sum()
		if(trainy[i]==0):
			d0.append(temp)
		else:
			d1.append(temp)
	d0.sort()
	d1.sort()
	distance0.append(d0)
	distance1.append(d1)
distance0=np.array(distance0)
distance1=np.array(distance1)
for i in range(1,51):
	distance0[:,i:i+1]=distance0[:,i:i+1]+distance0[:,i-1:i]
	distance1[:,i:i+1]=distance1[:,i:i+1]+distance1[:,i-1:i]
for i in range(0,50):
	sum=0
	for j in range(len(remainx)):
		if(distance0[j][i]>distance1[j][i]):
			if(remainy[j]==1):
				sum=sum+1
		else:
			if(remainy[j]==0):
				sum=sum+1
	print "k = ",i+1," Accuracy = ",(sum*1.0)/len(remainy)

