import sys
from Utils import Utils
import numpy as np
from sklearn.linear_model import LogisticRegression

def sigmoid(beta, X): # logistic function
    return 1.0/(1 + np.exp(-np.dot(X, beta.T)))

def gradientBeta(beta, X, y): # dL/dbeta
    a = sigmoid(beta, X)
    part1 = a - y.reshape(X.shape[0], 1)
    grad = np.dot(part1.T, X)
    return grad

def logLoss(beta, X, y):
    a = sigmoid(beta, X) # actual output
    loss = -(y * np.log(a) + (1 - y) * np.log(1 - a))
    return np.sum(loss)

def trainUsingGradientDescent(X, y, beta, num_iter, alpha = .01):
    loss = logLoss(beta, X, y)
    for i in range (num_iter):
        beta = beta - (alpha * gradientBeta(beta, X, y))
        loss = logLoss(beta, X, y)
        if (i%10 == 0):
            print('iter = ' + str(i) + ' loss=' + str(loss))
    return beta

def classify_data(beta, X): # 0 or 1
    a = sigmoid(beta, X) # actual output
    decision = np.where(a >= .5, 1, 0)
    return decision

def main():
    utils = Utils()
    #data = utils.readData('D:\DATA MINING\DataSet.csv') # load data
    data = utils.readDataRandom() # load data
    X = utils.normalize_data(data[:,0:2]) # or [:,:-1] normalize data - scale between 0-1
    X = np.hstack((np.ones((1,X.shape[0])).T, X)) # add 1's column to data
    Y = data[:, -1] # expected output, -1 means last column
    beta = np.zeros((1,X.shape[1])) # (1,3) in this example
    beta = trainUsingGradientDescent(X, Y, beta,1000) # optimize using gradient descent
    print("Logistic Regression Model coefficients:", beta)
    y_predicted = classify_data(beta, X) # predictions by the trained model
    print("Number of correct predictions = ", str(np.sum(Y ==
    y_predicted.reshape(Y.shape[0]))/len(X)*100) + '%')
    utils.plot_result(X, Y, beta) # plot results


    #using skleran 
    #utils = Utils()
    #data = utils.readDataRandom() # load data

    #X = data[:,0:2] # use only two features, for visualization purposes
    #Y = data[:, -1] # expected output, -1 means last column
    #model = LogisticRegression(solver='lbfgs')
    #model.fit(X, Y)
    #preds = model.predict(X) # predictions by the trained model
    #accuracy = (preds == Y).mean()
    #print("Number of correct predictions = ", str(np.sum(Y == preds.reshape(Y.shape[0]))/len(X)*100) + '%')
    #print("Accuracy SKL= " + str(accuracy))
    #print(model.intercept_, model.coef_)




if __name__ == "__main__":
    sys.exit(int(main() or 0))

