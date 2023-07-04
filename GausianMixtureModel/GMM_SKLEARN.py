import sklearn
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy import stats
import sys
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt 

def main():
	#---------------Iris Dataset-------------
	#df = pd.read_csv("./car data.csv")
	df = pd.read_csv("./iris.csv")
	
	dfrandom = df 
	
	#df1 = dfrandom.iloc[:,0:6].astype(int)
	df1 = dfrandom.iloc[:,0:4].astype(int)
	#---separate out the last column
	#df2 = dfrandom.iloc[:,6]
	df2 = dfrandom.iloc[:,4]
	
	dfrandom = pd.concat([df1,df2],axis=1)
	print(dfrandom)
	#--------------GMM-----------------
	#xdata = df1.values[:,0:6]
	xdata = df1.values[:,0:4]
	

	print(xdata.shape)
	N = xdata.shape[0]
	#gm = GaussianMixture(n_components=4, max_iter=50)
	gm = GaussianMixture(n_components=3, max_iter=20)
	gm.fit(xdata)
	print(gm.means_)
	print(gm.weights_)
	print(gm.covariances_)
	z = gm.score(xdata)
	print(z)

	#-----compute accuracy--------
	preds = gm.predict(xdata)
	print(preds)

	cluster_assigned = []
	acc = 0
	cluster_assigned = [mode(preds[0:50])[0], mode(preds[50:100])[0], mode(preds[100:150])[0]]
	 
	for i in range(0,N):
		if preds[i] == cluster_assigned[0] and i < 50: # first 50 members belong to class 0 
			acc = acc + 1
		if preds[i] == cluster_assigned[1] and i >=50 and i < 100: # next 50 are class 1
			acc = acc + 1
		if preds[i] == cluster_assigned[2] and i >= 100 and i < 150: # last 50, class 2
			acc = acc + 1
	# since GMM is unsupervised, class assignments to clusters may vary on each run
	#cluster_assigned = [mode(preds[0:400])[0], mode(preds[400:800])[0], mode(preds[800:1200])[0] , mode(preds[1200:1728])[0]]
	
	#for i in range(0,N):
	#	if preds[i] == cluster_assigned[0] and i < 400: # first 432 members belong to class 0
	#		acc = acc + 1
	#	if preds[i] == cluster_assigned[1] and i >=400 and i < 800: # next 432 are class 1
	#		acc = acc + 1
	#	if preds[i] == cluster_assigned[2] and i >= 800 and i < 1200: # next 432 , class 2
	#		acc = acc + 1
	#	if preds[i] == cluster_assigned[3] and i >= 1200 and i < 1728: # last 432, class 3
	#		acc = acc + 1

	print('accuracy =',acc/N*100)
if __name__ == "__main__":
	   sys.exit(int(main() or 0))