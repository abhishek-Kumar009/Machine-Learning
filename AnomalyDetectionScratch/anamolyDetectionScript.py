import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import math

dataset = sio.loadmat('anomalyData.mat')
X = dataset['X']
Xval = dataset['Xval']
yval = dataset['yval']

plt.scatter(X[:, 0], X[:, 1], marker = "x")
plt.xlabel('Latency(ms)')
plt.ylabel('Throughput(mb/s)')

def estimateGaussian(X):
    n = np.size(X, 1)
    m = np.size(X, 0)
    mu = np.zeros((n, 1))
    sigma2 = np.zeros((n, 1))
    
    mu = np.reshape((1/m)*np.sum(X, 0), (1, n))
    sigma2 = np.reshape((1/m)*np.sum(np.power((X - mu),2), 0),(1, n))
    
    return mu, sigma2

mu, sigma2 = estimateGaussian(X)

def multivariateGaussian(X, mu, sigma2):
     n = np.size(sigma2, 1)
     m = np.size(sigma2, 0)
     #print(m,n)
     
     if n == 1 or m == 1:
        # print('Yes!')
         sigma2 = np.diag(sigma2[0, :])
     #print(sigma2)
     X = X - mu
     pi = math.pi
     det = np.linalg.det(sigma2)
     inv = np.linalg.inv(sigma2)
     val = np.reshape((-0.5)*np.sum(np.multiply((X@inv),X), 1),(np.size(X, 0), 1))
     #print(val.shape)
     p = np.power(2*pi, -n/2)*np.power(det, -0.5)*np.exp(val)
     
     return p
 
p = multivariateGaussian(X, mu, sigma2)
#print('\n\nsome values of P are:',p[1],p[23],p[45],p.shape)

# =========== Working out for threshHold e ===================

pval = multivariateGaussian(Xval, mu, sigma2)

def selectThreshHold(yval, pval):
    
    F1 = 0
    bestF1 = 0
    bestEpsilon = 0
    
    stepsize = (np.max(p) - np.min(p))/1000
        
    epsVec = np.arange(np.min(p), np.max(p), stepsize)
    noe = len(epsVec)
    
    for eps in range(noe):
        epsilon = epsVec[eps]
        pred = (pval < epsilon)
        tp,fp,fn = 0,0,0
        for i in range(len(pval)):
            if pred[i] == 1 and yval[i] == 1:
                tp+=1
            elif pred[i] == 1 and yval[i] == 0:
                fp+=1
            elif pred[i] == 0 and yval[i] == 1:
                fn+=1       
        
        prec = tp/(tp + fp)
        rec = tp/(tp + fn)
        
        F1 = 2*prec*rec/(prec + rec)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestF1, bestEpsilon

F1, epsilon = selectThreshHold(yval, pval)
print('Epsilon and F1 are:',epsilon, F1)

            
        
     
        
        
         
    
    