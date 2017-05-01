from math import log
import numpy as np
import csv
class Tree(object):
	def __init__(self,dataset,remdiscrete,remcontinuous):
		self.dataset=dataset
		self.remdiscrete=remdiscrete
		self.remcontinuous=remcontinuous
		self.discrete=True
		self.variableidx=-1
		self.median=None
		self.children={}
		self.train()
	def train(self):
		if len(set(i[-1] for i in self.dataset))==1:
			return None
		if self.remdiscrete==set() and self.remcontinuous==set():
			return None
		if not len(self.dataset):
			return None
		minval=float("inf")
		minidx=-1
		median=None
		for i in self.remdiscrete:
			cnt={}
			cntpositive={}
			for j in range(len(self.dataset)):
				if self.dataset[j][i] in cnt:
					cnt[self.dataset[j][i]]+=1
				else:
					cnt[self.dataset[j][i]]=1
			for j in cnt.keys():
				cntpositive[j]=0
			for j in range(len(self.dataset)):
				cntpositive[self.dataset[j][i]]+=self.dataset[j][-1]
			gini=0
			for j in cnt.keys():
				if cntpositive[j]!=0 and cntpositive[j]!=cnt[j]:
					gini-=(1.0*cnt[j]/len(self.dataset))*(1.0*cntpositive[j]/cnt[j])*log((1.0*cntpositive[j]/cnt[j]))+(1.0*cnt[j]/len(self.dataset))*(1-1.0*cntpositive[j]/cnt[j])*log((1-1.0*cntpositive[j]/cnt[j]))
			if gini<minval:
				minval=gini
				minidx=i
		for i in self.remcontinuous:
			cnt=[0,0]
			cntpositive=[0,0]
			med=np.median(self.dataset.T[i])
			
			for j in range(len(self.dataset)):
				cnt[self.dataset[j][i]>med]+=1
				cntpositive[self.dataset[j][i]>med]+=self.dataset[j][-1]
			gini=0
			for j in range(2):
				if cntpositive[j]!=0 and cntpositive[j]!=cnt[j]:
					gini-=(1.0*cnt[j]/len(self.dataset))*(1.0*cntpositive[j]/cnt[j])*log((1.0*cntpositive[j]/cnt[j]))+(1.0*cnt[j]/len(self.dataset))*(1-1.0*cntpositive[j]/cnt[j])*log((1-1.0*cntpositive[j]/cnt[j]))
			if gini<minval:
				minval=gini
				minidx=i
				median=med
		self.variableidx=minidx
		if minidx in self.remdiscrete:
			self.discrete=True
			mymap={}
			for j in range(len(self.dataset)):
				if self.dataset[j][minidx] in mymap:
					mymap[self.dataset[j][minidx]].append(self.dataset[j])
				else:
					mymap[self.dataset[j][minidx]]=[self.dataset[j]]
			for key,vals in mymap.iteritems():
				self.children[key]=Tree(np.array(vals),self.remdiscrete-{minidx},self.remcontinuous);
		if minidx in self.remcontinuous:
			self.discrete=False
			self.median=median
			mymap=[[],[]]
			for j in range(len(self.dataset)):
				mymap[self.dataset[j][minidx]>median].append(self.dataset[j])
			for key in range(2):
				self.children[key]=Tree(np.array(mymap[key]),self.remdiscrete,self.remcontinuous-{minidx});
	def test(self,x):
		if self.children=={}:
			return self.dataset[0][-1]
		if x[self.variableidx] in self.children:
			if self.discrete:
				return self.children[x[self.variableidx]].test(x)
			else:
				return self.children[x[self.variableidx]>self.median].test(x)
		else:
			cnt1=0
			cnt0=0
			for i in self.dataset:
				if i[-1]==0:
					cnt0+=1
				else:
					cnt1+=1
			if cnt0>cnt1:
				return 0
			else:
				return 1

csv_file_object = csv.reader(open('csvdataset.csv', 'rb'))
data=[]
for row in csv_file_object:
	data.append(row)
data=np.array(data)
data=data[2::].astype(np.float64)
data=data[:,1:]
traindata=data[0:21000,:]
testdata=data[21000:,:]
discrete={1,2,3,5,6,7,8,9,10}
continuous={0,4,11,12,13,14,15,16,17,18,19,20,21,22}
model=Tree(traindata,discrete,continuous)
ans=[]
for i in testdata:
	ans.append(model.test(i))
from sklearn.metrics import classification_report
print classification_report(ans,testdata[:,-1])
sum=0
for i in range(len(testdata)):
	if(ans[i]==testdata[i][-1]):
		sum=sum+1
print "Accuracy is "+str((sum*1.0)/len(ans))