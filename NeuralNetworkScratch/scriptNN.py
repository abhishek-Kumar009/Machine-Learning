import numpy as np
from scipy import optimize as op
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import time

database = sio.loadmat('digitRecog.mat')
X = database['X']
y = database['y']


wtsNN = sio.loadmat('wts.mat')
theta1 = wtsNN['Theta1']
theta2 = wtsNN['Theta2']
nn_params = np.hstack((theta1.flatten(), theta2.flatten()))

input_layer_size = 400
hidden_layer_size = 25
num_of_labels = 10

lamda = 1
args = (input_layer_size, hidden_layer_size, num_of_labels, X, y, lamda)
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def costFunc(nn_params, *args):
    theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size + 1)],
                                  (hidden_layer_size, input_layer_size + 1))
    theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size + 1):],
                                  (num_of_labels, hidden_layer_size + 1))
    m = len(y)
    
    Y = pd.get_dummies(y.flatten())
    
    ones = np.ones((m, 1))
    A1 = np.hstack((ones, X))
    A2 = sigmoid(A1@theta1.T)
    A2 = np.hstack((ones, A2))
    H = sigmoid(A2@theta2.T)
    
    t1 = np.multiply(Y, np.log(H))
    t2 = np.multiply(1 - Y, np.log(1 - H))
    st = t1 + t2      
    newSum = (1/(-m))*np.sum(st)    
    snew = np.sum(newSum)
    
    r1 = np.sum(np.sum(np.power(theta1[:, 1:], 2), axis = 1))
    r2 = np.sum(np.sum(np.power(theta2[:, 1:], 2), axis = 1))
    sr = (lamda/(2*m))*(r1 + r2)
    
    J = snew + sr
    return J

cost = costFunc(nn_params,input_layer_size, hidden_layer_size,
                num_of_labels, X, y, lamda)
print('Cost at the given weights \n{}'.format(cost))

def randInitialize(L_in, L_out):
    W = np.zeros((L_out, L_in + 1))    
    W = np.random.rand(L_out, L_in + 1)*2*np.sqrt(2/L_in)
    return W

thetar1 = randInitialize(input_layer_size, hidden_layer_size)
thetar2 = randInitialize(hidden_layer_size, num_of_labels)

nn_ini_params = np.hstack((thetar1.flatten(), thetar2.flatten()))

def gradientNN(nn_params, *args):
    
    
    theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size + 1)],
                                  (hidden_layer_size, input_layer_size + 1))
    theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size + 1):],
                                  (num_of_labels, hidden_layer_size + 1))
    m = len(y)
    
    Y = pd.get_dummies(y.flatten())
    Theta1grad = np.zeros(theta1.shape)
    Theta2grad = np.zeros(theta2.shape)
    
    ones = np.ones((m, 1))
    A1 = np.hstack((ones, X))
    A2 = sigmoid(A1@theta1.T)
    A2 = np.hstack((ones, A2))
    H = sigmoid(A2@theta2.T)
    
    delta3 = H - Y
    delta2 = np.multiply((delta3@theta2), A2)
    delta2 = np.array(delta2)
    delta2 = delta2[:, 1:]
    
    Delta1 = delta2.T@A1
    Delta2 = delta3.T@A2
    
    r1 = np.hstack((np.zeros((hidden_layer_size, 1)), theta1[:, 1:]))
    r2 = np.hstack((np.zeros((num_of_labels, 1)), theta2[:, 1:]))
    
    Theta1grad = np.array(Delta1/m + (lamda/m)*r1)
    Theta2grad = np.array(Delta2/m + (lamda/m)*r2)
    
    nn_backprop_para = np.hstack((Theta1grad.flatten(), Theta2grad.flatten()))
    return nn_backprop_para

def checkNNgradients(nn_ini_params, nn_backprop_para, input_layer_size,
                     hidden_layer_size, num_of_labels, cX, cy, clamda):
    initialPara = nn_ini_params
    back_prop = nn_backprop_para
    
    eps = 0.0001
    l = len(initialPara)
    
    
    for i in range(10):
        rnum = int(np.random.rand()*l) 
        epsilon = np.zeros((l, 1))
        epsilon[rnum] = eps
        
        thetaPlus = initialPara.flatten() + epsilon.flatten()
        thetaMinus = initialPara.flatten() - epsilon.flatten()
        
        costPlus = costFunc(thetaPlus, input_layer_size, hidden_layer_size,
                            num_of_labels, cX, cy, clamda)
        costMinus = costFunc(thetaMinus, input_layer_size, hidden_layer_size,
                            num_of_labels, cX, cy, clamda)
        
        numGrad = (costPlus - costMinus)/(2*eps)
        print("Element: {0}. Numerical_Gradient = {1:.9f}. Actual_Gradient = {2:.9f}".
              format(rnum, numGrad, back_prop[rnum]))

nn_backprop_para = gradientNN(nn_ini_params, input_layer_size, hidden_layer_size, 
                              num_of_labels, X, y, lamda)
checkNNgradients(nn_ini_params, nn_backprop_para,input_layer_size, hidden_layer_size, 
                 num_of_labels, X, y, lamda)

# ============= Optimization =======================

ThetaBest = op.fmin_cg(costFunc, nn_ini_params, fprime = gradientNN,
                       args = (input_layer_size, hidden_layer_size, 
                               num_of_labels, X, y, lamda), maxiter = 200)

def predictionNN(ThetaBest, input_layer_size, hidden_layer_size, 
                 num_of_labels, X,y,lamda):
    
    theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size + 1)],
                                  (hidden_layer_size, input_layer_size + 1))
    theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size + 1):],
                                  (num_of_labels, hidden_layer_size + 1))
    m = len(y)
    ones = np.ones((m, 1))
    A1 = np.hstack((ones, X))
    A2 = sigmoid(A1@theta1.T)
    A2 = np.hstack((ones, A2))
    H = sigmoid(A2@theta2.T)
    
    predVec = np.argmax(H, axis = 1) + 1
    
    return predVec

'''def displayData(trainEg):
    fig, axes = plt.subplots(1,1)
    axes.imshow(trainEg.reshape(20, 20),cmap = "hot")
    '''
    

'''def predictLive(ThetaBest, X):    
    
    m = np.size(X, 0)
    
    randNum = np.random.permutation(m)
    for i in range(m):
        trainEg = X[randNum[i], :]
        prediction = predictionNN(ThetaBest,input_layer_size, hidden_layer_size, 
                                  num_of_labels, trainEg)
        displayData(trainEg)
        
        print("Prediction {}".format(prediction))
        time.sleep(5)'''
        
        
    

predictions = predictionNN(ThetaBest,input_layer_size, hidden_layer_size,
                           num_of_labels, X,y, lamda)
Accuracy = np.mean(predictions.flatten() == y.flatten())*100
print("Accuracy: {}".format(Accuracy))

'''print('Predicting live...\n')
predictLive(ThetaBest, X)'''

    
    

        
        
        
        
    
    
    
    
    


    
    
    
    
    
    
    

