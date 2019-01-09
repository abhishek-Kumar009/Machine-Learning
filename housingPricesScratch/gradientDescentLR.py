import numpy as np
import matplotlib.pyplot as plt
import costFunction as cf
def gradDesc(X, y, theta, learnRate, num_of_iter):
    m = len(X)
    X = np.c_[np.ones((m,1)), X]    
    J_history = np.zeros((num_of_iter,1))
    for i in range(num_of_iter):
        theta = theta - (learnRate/m)*((X.T).dot((X.dot(theta) - y)))
        J_history[i] = cf.costFunc(X,y,theta, 0)
        #print(J_history[i])
    plt.scatter(J_history,np.arange(num_of_iter),c = 'red')
    plt.xlabel("number of iterations")
    plt.ylabel("Cost Function")    
    plt.show()
    print("Final Cost:",J_history[-1])
    return theta
