import csv,numpy as np,random,math

csv_file_object = csv.reader(open('csvdataset2.csv', 'rb'))

data=[]
for row in csv_file_object:
	data.append(row)

data=np.array(data)
data=data[1::].astype(np.float64)
x=data[:,0:len(data[0])-1]
y=data[:,len(data[0])-1]

data=np.array(data)
data=data[1::].astype(np.float64)
x=data[:,0:len(data[0])-1]
y=data[:,len(data[0])-1]

x1=x[y==1]
temp=x[y==0]
np.random.shuffle(temp)
x0=temp[:900]
x=np.concatenate((x1,x0),axis=0)
y=np.array([1]*len(x1)+[0]*len(x0))

y=y.reshape((1392,1))

data=np.concatenate((x,y),axis=1)

np.random.shuffle(data)

with open("my_dataset.csv",'wb') as csvfile:
    my_writer=csv.writer(csvfile,delimiter=',')
    for i in data:
        my_writer.writerow(i)