
import sys
import Utils
import numpy as np
import pandas as pd

import math 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



def main():

 

        datafile = "D:/DATA MINING/Cancer dataset/data.csv"
        labels_file = "D:/DATA MINING/Cancer dataset/labels.csv"

        data = pd.read_csv(datafile, header=None)
        X = data.iloc[1: ,1: ].astype(float)


        label_data = pd.read_csv(labels_file, header=None)
        y = label_data.iloc[1: ,1: ].astype(int)


        #data = np.genfromtxt(
        #datafile,
        #delimiter=",",
        #usecols=range(1, 20532),
        #skip_header=1
        #)

        #true_label_names = np.genfromtxt(
        #labels_file,
        #delimiter=",",
        #usecols=(1,),
        #skip_header=1,
        #dtype="int"
        #)

        
               

        
        X = X.values 
        print(X.shape)
       
        y = y.values 
        ##------test KNN on an unknown data point----------

        X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=1192)
       
        knn_model = KNeighborsClassifier(n_neighbors=3)
        knn_model.fit(X_train, y_train)

        
        test_preds = knn_model.predict(X_test)
        acc=0
        for i in range (0,len(X_test)) :
            if test_preds[i] == y_test[i] :
                 acc = acc + 1 

        
        print(' Classification accuracy =',acc/len(X_test)*100)
        
if __name__ == "__main__":
    sys.exit(int(main() or 0))