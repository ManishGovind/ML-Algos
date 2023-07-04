
import sys
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from Utils import Utils
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv(r"C:\Users\Lenovo\Desktop\CPSC552\iris.csv")
    #---randomize data
    dfrandom = df #df.sample(frac=1, random_state=1119).reset_index(drop=True)
    # data read from a file is read as a string, so convert the first 4 cols to float
    df1 = dfrandom.iloc[:,0:4].astype(float)
    X = df1.values
    #---separate out the last column
    df2 = dfrandom.iloc[:,4]
    y = df2.values
    #--------
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(df2)  #'setosa'=0,'versicolor'=1,'virginica'=2

    # Creating Train and Test datasets
    X_train, X_test, y_train, y_test = train_test_split(X,y_enc, random_state = 50,test_size = 0.3)

    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    tree.plot_tree(clf)
    plt.show()
    # Predict Accuracy Score
    y_pred = clf.predict(X_test)
    print("Train data accuracy:",accuracy_score(y_true = y_train,y_pred=clf.predict(X_train)))
    print("Test data accuracy:",accuracy_score(y_true = y_test, y_pred=y_pred))
    # To be able to visualize data, lets reduce the dimensionality of feature space
    # from 4 to 2
   
    x_train_scaled = StandardScaler().fit_transform(X_train)
    x_test_scaled = StandardScaler().fit_transform(X_test)
    pca = PCA(n_components=2)
    principalComponents_train = pca.fit_transform(x_train_scaled)
    principalComponents_test = pca.fit_transform(x_test_scaled)
    pca_train_df = pd.DataFrame(data = principalComponents_train, columns = ['principal component 1', 'principal component 2'])
    pca_test_df = pd.DataFrame(data = principalComponents_test, columns = ['principal component 1', 'principal component 2'])
    clf_pca = DecisionTreeClassifier()
    clf_pca.fit(pca_train_df.values,y_train) # train decision tree on train data
    y_pred_pca = clf_pca.predict(pca_test_df.values)
    print("Test data accuracy after PCA:",accuracy_score(y_true = y_test,y_pred=y_pred_pca))

    Utils.visualize_classifier(clf_pca,pca_test_df.values, y_test, ax=None,cmap='rainbow')
    #----------try random forest classifier after pca------------
    modelrf = RandomForestClassifier(n_estimators=100, random_state=0)
    modelrf.fit(pca_train_df.values,y_train)
    Utils.visualize_classifier(modelrf, pca_test_df.values, y_test)
    y_pred_rf = modelrf.predict(pca_test_df.values)
    print("Test data accuracy after Random Forest:",accuracy_score(y_true = y_test,y_pred=y_pred_rf))
if __name__ == "__main__":
    sys.exit(int(main() or 0))