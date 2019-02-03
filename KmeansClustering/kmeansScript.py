import numpy as np
import scipy.io as sio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

dataKmeans = sio.loadmat('kmeansData.mat')
X = dataKmeans['X']

K = 3
centroids = np.array([[3,3],[6,2],[8,5]])

def findClosestCentroids(X, centroids):
    m = np.size(X, 0)
    K = np.size(centroids, 0)
    idx = np.zeros((m), dtype = int)
    for i in range(m):
        idx [i] = np.argmin(np.sum(np.power((centroids - X[i, :]), 2), 1))
    return idx

Xtmp = X[0:3,:]
idx = findClosestCentroids(X, centroids)
print('\n\nClosest centroids are:\n',idx[0:3])

def computeCentroids(idx, X, K):
    n = np.size(X, 1)
    m = np.size(X, 0)
    centroids = np.zeros((K,n))
    for i in range(K):
        chckVec = np.zeros((m))
        chckVec = (idx == i)
        chckVec = np.array([chckVec])
        count = np.count_nonzero(chckVec)
        multVct = (1/count)*(np.sum((np.multiply(X,chckVec.T)), 0))
        centroids[i] = multVct
    return centroids


centroids = computeCentroids(idx, X, K)
print('Computed Centroids are: \n\n',centroids)

max_iter = 10
initial_centroids = np.array([[3,3],[6,2],[8,5]])

def runKmeans(X, initial_centroids, max_iterations):    
    K = np.size(initial_centroids, 0)
    centroids = initial_centroids
    
    for i in range(max_iter):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(idx, X, K)
    return idx, centroids

idx, centroids = runKmeans(X,initial_centroids, max_iter)
print('\n\n Centroids after K-means are: \n',centroids)

print('\n\n Running K-means on image...\n')

img = mpimg.imread('bird_small.png')
imgm = np.size(img, 0)
imgn = np.size(img, 1)
imgMod = np.reshape(img, (imgm*imgn, 3))

K = 50
max_iter = 10

def kmeansInitCentroids(X, K):
    n = np.size(X, 1)
    m = np.size(X, 0)
    centroids = np.zeros((K, n))
    randperm = np.random.permutation(m)
    centroids = X[randperm[:K], :]
    return centroids

ini_centroids = kmeansInitCentroids(imgMod, K)

idx, centroids = runKmeans(imgMod, ini_centroids, max_iter)
print('\n\nFirst 3 centroids are', centroids[:3,:])

# ========= Compressing image by k-means ================

idx = findClosestCentroids(imgMod, centroids)


img_new = centroids[idx[0:], :]
img_recovered = np.reshape(img_new, (imgm, imgn, 3))
plt.imshow(img_recovered)

    
    
        
    

        