import sys 
import numpy as np 

def compute_svd(A):

    u,s,v = np.linalg.svd(A)
    print(u)
    print("----u----")
    
    print(v)
    print("----v----")
    s = np.diag(s)
    if (s.shape[1] != A.shape[1]):
    # stack zero columns in the diagonal matrix
        num_zero_cols = A.shape[1] - s.shape[1]
        sz = np.zeros((s.shape[0],A.shape[1]))
        sz[:,:-1] = s # extra columns of zeros
        s = sz
    print(s)
    print("----s----")
    Asvd = np.dot(np.dot(u,s),v)
    print("----A from SVD components------")
    print(Asvd)

def main():
    # compute Eigen values, Eigen Vectors, SVD for
    #[[3,1],
    # [1,3]]
    Alist = [[2,1],[1,2]]
    A = np.asarray(Alist, dtype=float) # convert list to numpy array
    eigen_vals = np.linalg.eigvals(A)

    eigenvs, eigen_vecs = np.linalg.eig(A)
   
    #compute_svd(A)
    # compute SVD of [[3,2,2],[2,3,-2]]
   # print("-----------second example - SVD of 2x3 matrix")
    A2list = [[3,2,2],[2,3,-2]]
    A2 = np.asarray(A2list, dtype=float)
    compute_svd(A2)
if __name__ == "__main__":
    sys.exit(int(main() or 0))


