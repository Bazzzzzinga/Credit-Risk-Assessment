import os,csv,math
import numpy as np
from sklearn.metrics import classification_report

#Reading Data
csv_file_object = csv.reader(open('modcsvdataset.csv', 'rb'))
data=[]
for row in csv_file_object:
	data.append(row)

data=np.array(data)
data=data[:,1:len(data[0])]

f=open("train.csv","w")
for i in range(21000):
	for j in range(len(data[0])):
		f.write(data[i][j])
		if j!=len(data[0])-1:
			f.write(",")
	f.write('\n')
f.close()

f=open("test.csv","w")
for i in range(21000,30000):
	for j in range(len(data[0])):
		f.write(data[i][j])
		if j!=len(data[0])-1:
			f.write(",")
	f.write('\n')
f.close()
