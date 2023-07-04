import numpy as np 
 
from DistanceMetric import * 
 
class PCA(object):  # column wise data computations 
    def __init__(self): 
        self.projections = [] 
        self.dist_metric = EuclideanDistance() 
 
    def asRowMatrix(self,X):  
        if len(X) == 0:  
            return np.array([])  
        mat = np.empty((0, X[0].size), dtype=X[0].dtype)  
         
        for row in X:  
            mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))  
        return mat 
 
    def asColumnMatrix(self,X):  
        if len(X) == 0:  
            return np.array([])  
        mat = np.empty((X[0].size , 0), dtype=X[0].dtype)  
        for col in X:  
            mat = np.hstack((mat, np.asarray(col).reshape(-1,1)))  
        return mat 
 
    def pca_col(self,X, num_components=0):  
        [d,n] = X.shape   # n - number of images, d = input dimension e.g. 112x92=10304 
        if (num_components <= 0) or (num_components > n):  
            num_components = n  
        mu = X.mean(axis=1)                   #mu= X.mean(axis=0)row matrix
        X = (X.T - mu).T  
 
 
        if d<n:  
            Cov = np.dot(X,X.T) # covariance matrix 
            [eigenvalues , EV] = np.linalg.eigh(Cov)  
        else:  
            Cov = np.dot(X.T,X)  
            [eigenvalues ,eigenvectors] = np.linalg.eigh(Cov)  
            EV = np.dot(X,eigenvectors) # it is termed as EigenFace  
         
        # convert each eigenvector to a unit vector by dividing it by its norm 
        for i in range(n):  
            EV[:,i] = EV[:,i]/np.linalg.norm(EV[:,i])  
 
        #  sort eigenvectors descending by their eigenvalue  
        idx = np.argsort(-eigenvalues)  
        eigenvalues = eigenvalues[idx]  
        EV = EV[:,idx]  
        #  select only num_components  
        eigenvalues = eigenvalues[0:num_components].copy()  
        EV = EV[:,0:num_components].copy()  
        return [eigenvalues , EV , mu] 
 

    def pca_row(self,X, num_components=0):  
        [d,n] = X.shape   # n - number of images, d = input dimension e.g. 112x92=10304 
        if (num_components <= 0) or (num_components > n):  
            num_components = n  
        mu = X.mean(axis=0)                   #mu= X.mean(axis=0)row matrix
        X = (X - mu) 
 
 
        if n>d:
            C = np.dot(X.T,X)
            [eigenvalues ,EV] = np.linalg.eigh(C)
        else:
            C = np.dot(X,X.T)
            [eigenvalues ,eigenvectors] = np.linalg.eigh(C)
            EV = np.dot(X.T,eigenvectors)
        for i in range(n):
            EV[:,i] = EV[:,i]/np.linalg.norm(EV[:,i]) 
 
        #  sort eigenvectors descending by their eigenvalue  
        idx = np.argsort(-eigenvalues)  
        eigenvalues = eigenvalues[idx]  
        EV = EV[:,idx]  
        #  select only num_components  
        eigenvalues = eigenvalues[0:num_components].copy()  
        EV = EV[:,0:num_components].copy()  
        return [eigenvalues , EV , mu] 
 
    

    def project(self, EV, X, typ,mu=None ):
        if typ == "row": 
            if mu is None:
                return np.dot(X,EV)
            return np.dot(X - mu, EV)    
        elif typ == "col":
            if mu is None:  
                 return np.dot(EV.T,X) 
            return np.dot(EV.T,(X.T - mu).T).T

 
    def reconstruct(self, EV, Y, typ,mu=None ): #
        if typ == "row": 
            if mu is None:
                return np.dot(Y,EV.T)
            return np.dot(Y, EV.T) + mu   
        elif typ == "col":
            if mu is None:  
                return np.dot(EV,Y.T)  
            return (np.dot(EV,Y.T).T + mu).T
        
    

   
         
     
    def predict( self,EF, X, mu, y, yseq , typ):  
        minDist = np.finfo('float').max  
        minClass = -1 
        index = -1 
        if typ == "col":
            Q = self.project(EF, X.reshape(-1,1), typ, mu).T  
        else:
            Q = self.project(EF, X.reshape(-1,1),typ, mu)  
        for i in range(len(self.projections)):  
            dist = self.dist_metric(self.projections[i], Q)  
            if dist < minDist:  
                minDist = dist  
                minClass = y[i] 
                index = yseq[i] 
        return minClass, index 




 
    
 
    

