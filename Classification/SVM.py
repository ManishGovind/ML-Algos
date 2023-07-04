
import sys 
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import itertools
import pandas as pd
import matplotlib


class IrisSVM(object):
    def __init__(self) -> None:
        df = pd.read_csv(r"C:\Users\Lenovo\Desktop\CPSC552\iris.csv")
        df1 = df.iloc[:,0:4].astype(float)
        self.X = df1.to_numpy()
        #---separate out the last column
        df2 = df.iloc[:,4]
        self.y = df2.to_numpy()
    def make_meshgrid(self, x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h))
        return xx, yy
    def plot_contours(self, ax, model, xx, yy, **params):
        """Plot the decision boundaries for a classifier.
        Parameters
        ----------
        ax: matplotlib axes object
        model: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out
    def plotIrisDecisionBoundaries(self):
         iris = datasets.load_iris()
         # first two features for visualization
         X = iris.data[:, :2]
         y = iris.target
         C = 1.0 # SVM regularization parameter
         models = (svm.SVC(kernel='linear', C=C),
         svm.LinearSVC(C=C),
         svm.SVC(kernel='rbf', gamma='auto', C=C),
         svm.SVC(kernel='poly', gamma='auto', degree=3, C=C))
         models = (clf.fit(X, y) for clf in models)
         # title for the plots
         titles = ('SVC with linear kernel',
         'LinearSVC (linear kernel)',
        'SVC with RBF kernel',
        'SVC with polynomial (degree 3) kernel')
         # Set-up 2x2 grid for plotting.
         fig, sub = plt.subplots(2, 2)
         plt.subplots_adjust(wspace=0.4, hspace=0.4)
         X0, X1 = X[:, 0], X[:, 1]
         xx, yy = self.make_meshgrid(X0, X1)
         for clf, title, ax in zip(models, titles, sub.flatten()):
             self.plot_contours(ax, clf, xx, yy,
             cmap=plt.cm.coolwarm, alpha=0.8)
             ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
             ax.set_xlim(xx.min(), xx.max())
             ax.set_ylim(yy.min(), yy.max())
             ax.set_xlabel('Sepal length')
             ax.set_ylabel('Sepal width')
             ax.set_xticks(())
             ax.set_yticks(())
             ax.set_title(title)
         plt.show()

    def svmAccuracy(self):
         iris = datasets.load_iris()
         X = iris.data[:, :2] # first two features, gives around 82% accuracy
         X = iris.data[:, :4] # all 4 features, gives around 96-98% accuracy
         y = iris.target
         C = 1.0 # SVM regularization parameter
         models = (svm.SVC(kernel='linear', max_iter=4000, C=C),
         svm.LinearSVC(C=C, max_iter=4000),
         svm.SVC(kernel='rbf', gamma='auto', C=C),
         svm.SVC(kernel='poly', gamma='auto', degree=3, C=C))
         models = (model.fit(X, y) for model in models)
         for model in models:
             Z = model.predict(X)
         countCorrect = np.sum(Z==y)
         print("accuracy of " + str(model) + " = " + str (countCorrect/len(Z)))

    def visuvalize_sepal_data(self):
         iris = datasets.load_iris()
         X = iris.data[:, :2] # we only take the first two features.
         y = iris.target
         #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
         plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.winter)
         plt.xlabel('Sepal length')
         plt.ylabel('Sepal width')
         plt.title('Sepal Width & Length')
         plt.show()

    def visuvalize_petal_data(self):
        iris = datasets.load_iris()

        X = iris.data[:, 2:] # we only take the last two features.
        y = iris.target
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.winter)
        plt.xlabel('Petal length')
        plt.ylabel('Petal width')
        plt.title('Petal Width & Length')
        plt.show()

    def visualizeWithPCA(self):
        yy = np.zeros(self.X.shape[0])
        for i in range(len(yy)):
            if self.y[i] == 'setosa':
                yy[i] = 0
            if self.y[i] == 'versicolor':
                yy[i] = 1
            if self.y[i] == 'virginica':
                yy[i] = 2
        data = self.X # load data
        X = data[:,0:4]
        # plot the first three PCA dimensions
        fig = plt.figure(1, figsize=(8, 6))
        X_reduced = PCA(n_components=3).fit_transform(X)
        ax = plt.axes(projection ='3d')
        # defining axes
        x = X_reduced[:, 0]
        y = X_reduced[:, 1]
        z = X_reduced[:, 2]
        mycmap = matplotlib.colors.ListedColormap(["blue", "red","green"])
        ax.scatter(x, y, z, c = yy, cmap=mycmap)
        ax.set_title('PCA - IRIS (3-components)')
        plt.show()

    def svmWithPCAReducedFeatures(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        pca = PCA(n_components=3)
        pca.fit(X)
        X_reduced = pca.transform(X)
        C = 1.0 # SVM regularization parameter
        models = [svm.SVC(kernel='linear', max_iter=4000, C=C),
        svm.LinearSVC(C=C, max_iter=4000),
        svm.SVC(kernel='rbf', gamma='auto', C=C),
        svm.SVC(kernel='poly', gamma='auto', degree=3, C=C)]
        i = 0
        for model in models:
            models[i] = model.fit(X_reduced, y)
            i = i + 1
        print(models[0])
        i = 0
        for model in models:
            Z = model.predict(X_reduced)
            countCorrect = np.sum(Z==y)
            print("accuracy of model " + str(i) + " = " + str(countCorrect/len(Z)))
            i = i + 1
            X_r = pca.transform(X[125].reshape(1,4)) # predict class for data 125
    
            Z125 = model.predict(X_r)
            print("class for data 125 = " + str(Z125[0])) 




def main():
    irissvm = IrisSVM()
    irissvm.visuvalize_sepal_data()
    irissvm.visuvalize_petal_data()
    irissvm.visualizeWithPCA()
    irissvm.svmAccuracy()
    irissvm.plotIrisDecisionBoundaries()
    irissvm.svmWithPCAReducedFeatures()

if __name__ == "__main__":
 sys.exit(int(main() or 0))
