import numpy as np
def predict(X, thetaBest):
    m = len(X)
    X = np.c_[np.ones((m,1)), X] 
    hyp = X.dot(thetaBest)
    return hyp

