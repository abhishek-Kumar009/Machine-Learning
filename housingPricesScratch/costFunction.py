import numpy as np
import math
def costFunc(X, y, theta, lamda):
    result = 0
    # ========== LOOP METHOD ================
    '''for i in range(len(X)):
        #print('i = ',i)
        hyp = theta[0] + sum(X[i,:].dot(theta[1:]))  
       # print('Sqr error',hyp - y[i])
        result+= pow((hyp - y[i]),2)
        #print(result)
    return result*(1/(2*len(X)))'''
    # =======================================
    m = len(X)
    hyp = X.dot(theta)
    sqErr = sum(pow((hyp - y),2)) + (lamda/(2*m))*sum(pow(theta[1:], 2))
    result = (1/(2*m))*sqErr
    return result
    

  


    
    
    
