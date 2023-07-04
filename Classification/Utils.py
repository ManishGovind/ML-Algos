import csv
import numpy as np
import matplotlib.pyplot as plt




class Utils(object):
    def readData(self,filename): # return as numpy array
        with open(filename,"r") as csvfile:
            lines = csv.reader(csvfile)
            data = list(lines)
            for i in range(len(data)): # convert data to float
                data[i] = [float(x) for x in data[i]]
        return np.array(data)
    def readDataRandom(self):
        np.random.seed(12)
        num_observations = 50
        x1 = np.random.multivariate_normal([1.5, 4], [[1, .75],[.75, 1]],num_observations)
        print(x1.shape)
        x2 = np.random.multivariate_normal([1, 2], [[1, .75],[.75, 1]],num_observations)
        data = np.vstack((x1, x2)).astype(np.float32)
        print(data.shape)
        labels = np.hstack((np.zeros(num_observations),
        np.ones(num_observations)))
        print(labels.shape)
        dataWithLabels = np.hstack((data,labels.reshape(labels.shape[0],1)))
        #print(dataWithLabels.shape)
        #print(labels.shape)
        plt.figure(figsize=(12,8))
        plt.scatter(data[:, 0], data[:, 1], c = labels, alpha = .4)
        plt.show()
        return dataWithLabels
    def normalize_data(self,X):
        min = np.min(X, axis = 0)
        max = np.max(X, axis = 0)
        normX = 1 - ((max - X)/(max-min))
        return normX
    def plot_result(self,X, y, beta):
        x_0 = X[np.where(y == 0.0)]
        x_1 = X[np.where(y == 1.0)]
        # plot the data points
        plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='b', label='y = 0')
        plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1')
        # plot the decision boundary
        x1 = np.arange(0, 1, 0.1)
        x2 = -(beta[0,0] + beta[0,1]*x1)/beta[0,2]
        plt.plot(x1, x2, c='g', label='reg line')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.show()

    def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
        ax = ax or plt.gca()
        # Plot the training points
        ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,clim=(y.min(), y.max()), zorder=3)
        ax.axis('tight')
        ax.axis('off')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx, yy = np.meshgrid(np.linspace(*xlim, num=200),np.linspace(*ylim, num=200))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        # Create a color plot with the results
        n_classes = len(np.unique(y))
        contours = ax.contourf(xx, yy, Z, alpha=0.3,levels=np.arange(n_classes + 1) - 0.5,
                            cmap=cmap, clim=(y.min(), y.max()),zorder=1)
        ax.set(xlim=xlim, ylim=ylim)
        plt.show()
