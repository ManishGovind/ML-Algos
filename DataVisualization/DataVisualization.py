
import sys
from sklearn.datasets import load_digits
from sklearn.manifold import MDS , Isomap  , TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import Utils
import umap
import numpy as np
import matplotlib.pyplot as plt


def main():
    
    X, y = Utils.read_data() # cancer data
    # randomly select 800 samples from dataset
    np.random.seed(100)
    subsample_idc = np.random.choice(X.shape[0], 800, replace=False)
    X = X[subsample_idc,:]
    y = y[subsample_idc]
    y = np.array([int(lbl) for lbl in y])
    num_components = 2
    mds = MDS(n_components=num_components)
    X_reduced = mds.fit_transform(X)
    print(X_reduced.shape)
    plt.title('MDS')
    plt.scatter(X_reduced[:,0], X_reduced[:,1], s= 5, c=y, cmap='Spectral')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(len(np.unique(y))))
    
    plt.show()
    

    # isomap
    isomap = Isomap(n_components=num_components)
    X_reduced = isomap.fit_transform(X)
    print(X_reduced.shape)
    plt.title('ISOMAP')
    plt.scatter(X_reduced[:,0], X_reduced[:,1], s= 5, c=y, cmap='Spectral')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(len(np.unique(y))))
    

    plt.show()
    #PCA
    X = StandardScaler().fit_transform(X) # subtract mean, divide by var
    pca = PCA(num_components)
    pca_result = pca.fit_transform(X)
    print(pca_result.shape) # 2d for visualization
    plt.title('PCA')
    plt.scatter(pca_result[:,0], pca_result[:,1], s= 5, c=y, cmap='Spectral')
    plt.colorbar(boundaries=np.arange(11)-
    0.5).set_ticks(np.arange(len(np.unique(y))))
    plt.show() 

    #TSNE
    tsne = TSNE(num_components,perplexity=30)
    tsne_result = tsne.fit_transform(X)
    print(tsne_result.shape)
    plt.title('t-SNE')
    plt.scatter(tsne_result[:,0], tsne_result[:,1], s= 5, c=y, cmap='Spectral')
    plt.colorbar(boundaries=np.arange(11)- 0.5).set_ticks(np.arange(len(np.unique(y))))
    plt.show()

    #u-map
    ump = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    n_components=num_components,
    metric='euclidean'
    )
    umap_result = ump.fit_transform(X)
    print(umap_result.shape) # 2d for visualization
    plt.title('UMAP')
    plt.scatter(umap_result[:,0], umap_result[:,1], s= 5, c=y, cmap='Spectral')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(len(np.unique(y))))
    plt.show()


if __name__ == "__main__":
   sys.exit(int(main() or 0))




