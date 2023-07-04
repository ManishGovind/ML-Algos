
import sys
import pandas as pd 
import math
import numpy  as np 
import random
import csv





















# the categorical class names are changed to numberic data
def encode_class(dataframe):
	classes = []
	for i in range(len(dataframe)):
		if dataframe.iloc[i][-1] not in classes:
			classes.append(dataframe.iloc[i][-1])
	#for i in range(len(classes)):
	#	for j in range(len(dataframe)):
	#		if dataframe.iloc[j][-1] == classes[i]:
	#			dataframe.loc[[j],'species'] = i
	return dataframe , classes		
			

# Splitting the data
def splitting(df, ratio):
	
	train_data = df.sample(frac=ratio,random_state=200)
	test_data = df.drop(train_data.index)
	return train_data, test_data


# Group the data rows under each class yes or
# no in dictionary eg: dict[yes] and dict[no]
def groupUnderClass(data , classes):
	class_group_dict = {}
	class_header = str(data.columns[-1])
	for i in classes:
		class_group_dict[i] =  data[data[class_header] == i]
	return class_group_dict



def MeanAndStdDev(data):
	info = ( data.iloc[:,0:-1].mean(axis = 0 ) , data.iloc[:,0:-1].var(axis=0))
	return info

# find Mean and Standard Deviation under each class
def MeanAndStdDevForClass(data , classes):
	info = {}
	dict = groupUnderClass(data , classes)
	#print(dict[0],dict[2],dict[1])
	for classValue, instances in dict.items():		
		info[classValue] = MeanAndStdDev(instances)
	return info


# Calculate Gaussian Probability Density Function
def calculateGaussianProbability(x, mean, var):
	expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(var, 2))))
	return (1 / (math.sqrt(2 * math.pi) * var)) * expo



# Calculate Class Probabilities
def calculateClassProbabilities(info, test_item):
	probabilities = {}
	x = test_item.values
	for classValue, classSummaries in info.items():
		probabilities[classValue] = 1
		mean, std_dev = classSummaries
		for i in range(len(x)):
			probabilities[classValue] *= calculateGaussianProbability(x[i], mean.values[i], std_dev.values[i])
	return probabilities


# Make prediction - highest probability is the prediction
def predict(info, test):
	probabilities = calculateClassProbabilities(info, test)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

# returns predictions for a set of examples
def getPredictions(info, df_test):
	predictions = []
	
	for i in range(len(df_test)):
		result = predict(info, df_test.iloc[i][:-1]  )
		predictions.append(result)
	return predictions

# Accuracy score
def accuracy_rate(test, predictions):
	correct = 0
	for i in range(len(test)):
		if test.iloc[i][-1] == predictions[i]:
			correct += 1
	return (correct / float(len(test))) * 100.0




#read the dataset 
df = pd.read_csv("./wheat-seeds.csv")
#---randomize data

df = df.dropna()

df = df.drop_duplicates()

dfrandom = df.sample(frac=1, random_state=1119).reset_index(drop=True)
#  convert the data  first 4 cols to float
df1 = dfrandom.iloc[:,0:-1].astype(float)
#---separate out the last column
df2 = dfrandom.iloc[:,-1]

#---combine the 4 numerical columns and the ast column that has the flower category
dfrandom = pd.concat([df1,df2],axis=1)

mydata  , classes = encode_class(dfrandom)
print(classes)




	
split_ratio = 0.90
# 70% of data is training data and 30% is test data used for testing
#ratio = 0.7


train_data, test_data = splitting(mydata, split_ratio)

print('Total number of series are: ', len(mydata))
print('Out of these, training data are: ', len(train_data))
print("Test data is: ", len(test_data))

## prepare model
info = MeanAndStdDevForClass(train_data , classes)

## test model
predictions = getPredictions(info, test_data)
accuracy = accuracy_rate(test_data, predictions)
print("Accuracy of your model is: ", accuracy)












