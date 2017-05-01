import os,csv,math
import numpy as np
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
x=data[:,0:len(data[0])-1]
y=data[:,len(data[0])-1]
x=x[:,:].astype(np.float64)
y=y[:].astype(np.float64)
y.resize((y.shape[0],1))
trainx=x[0:int(0.7*len(data))]
trainy=y[0:int(0.7*len(data))]
remainx=x[int(0.7*len(data)):len(data)]
remainy=y[int(0.7*len(data)):len(data)]
data1=[]
data0=[]
for i in range(len(trainx)):
    if trainy[i]==1:
        data1.append(trainx[i])
    else:
        data0.append(trainx[i])
data1=np.array(data1)
data2=np.array(data0)
data1mean=np.mean(data1,axis=0)
data0mean=np.mean(data0,axis=0)
data1std=np.std(data1,axis=0)
data0std=np.std(data0,axis=0)
#print data1mean, data0mean, data1std, data0std
pr1=float(len(data1)*1.0/(len(data1)+len(data0)))
pr0=float(len(data0)*1.0/(len(data1)+len(data0)))
cnt=0
predicted=[]
for i in remainx:
        pro1=pr1
        pro0=pr0
        for j in range(len(i)):
              pro1=pro1*calculateprobability(i[j],data1mean[j],data1std[j])
              pro0=pro0*calculateprobability(i[j],data0mean[j],data0std[j])  
        if(pro0>pro1):
                predicted.append(0)
        else:
                predicted.append(1)
        cnt=cnt+1
predicted=np.array(predicted)
from sklearn.metrics import classification_report
sum=0
for i in range(len(predicted)):
    if(predicted[i]==remainy[i]):
        sum=sum+1
print "Accuracy is "+str((sum*1.0)/len(predicted))
print classification_report(predicted,remainy)

