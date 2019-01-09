import numpy as np
import statistics
def featureNorm(X):
    m = len(X)
    mu = np.zeros((m,1))
    sigma = np.zeros((m,1))
    for i in range(m):
        mu[i] = sum(X[i])/m
        X[i] = X[i] - mu[i]
    for j in range(m):
        sigma[j] = statistics.stdev(X[j])
        X[j] = X[j]/sigma[j]
    return X
    