import numpy as np
import scipy.io as sio
from scipy import optimize as op

movieParams = sio.loadmat('movieParameters.mat')
# print('keys:',movieParams.keys())

# loading the parameters Y and R
Y = movieParams['Y']
R = movieParams['R']
'''print(Y.shape)
print(R.shape)'''

print('Calculating average rating of all the films...\n')
multiMat = np.multiply(Y, R)
nonZeroEl = np.count_nonzero(multiMat, 1)
smMulti = np.sum(multiMat, 1)
avgRating = smMulti/nonZeroEl
print(avgRating)

movieWts = sio.loadmat('preTrainWts.mat')
# X,Theta, num_of_movies, num_of_users, num_of_features

# debugging cost function
num_of_movies = 5
num_of_users = 4
num_of_features = 3
X = movieWts['X']
Theta = movieWts['Theta']
Xtmp = X[0:num_of_movies, 0:num_of_features]
Thetatmp = Theta[0:num_of_users, 0:num_of_features]
Ytmp = Y[0:num_of_movies, 0:num_of_users]
Rtmp = R[0:num_of_movies, 0:num_of_users]
params = np.hstack((Xtmp.flatten(), Thetatmp.flatten()))
lamda = 100

def costFuncRec(params, Ytmp, Rtmp, num_of_users,
                num_of_movies, num_of_features, lamda):
    X = np.reshape(params[:num_of_movies*num_of_features],
                   (num_of_movies, num_of_features))
    Theta = np.reshape(params[num_of_movies*num_of_features:],
                       (num_of_users, num_of_features))
    hyp = X@Theta.T
    hypNew = np.multiply(hyp, Rtmp)
    Ynew = np.multiply(Ytmp, Rtmp)
    error = np.sum(np.sum(np.power((hypNew - Ynew), 2), 1))
    J = (1/2)*error
    reg_term = (lamda/2)*(np.sum(np.sum(np.power(Theta, 2), 1))) + (lamda/2)*(np.sum(np.sum(np.power(X, 2), 1)))
    return J + reg_term

cost = costFuncRec(params, Ytmp, Rtmp, num_of_users,
                   num_of_movies, num_of_features, 1.5)
print('Cost at the given weights is : {}'.format(cost))

def gradientsRec(params, Ytmp, Rtmp, num_of_users, 
                 num_of_movies, num_of_features, lamda):
    
    X = np.reshape(params[:num_of_movies*num_of_features],
                   (num_of_movies, num_of_features))
    Theta = np.reshape(params[num_of_movies*num_of_features:],
                       (num_of_users, num_of_features))
    
    hyp = X@Theta.T
    hypNew = np.multiply(hyp, Rtmp)
    Ynew = np.multiply(Ytmp, Rtmp)
    
    error = (hypNew - Ynew)
    
    X_grad = error@Theta + lamda*X
    Theta_grad = (error.T)@X + lamda*Theta
    return np.hstack((X_grad.flatten(),Theta_grad.flatten()))

def checkGradientsRec(initial_params, gradParams, Ytmp, Rtmp,
                      num_of_users, num_of_movies, num_of_features, lamda):
    l = len(initial_params)
    eps = 0.0001
    
    
    for i in range(10):
        epsilon = np.zeros((l))
        rnum = int(np.random.rand()*l)
        epsilon[rnum] = eps
        
        paramsPlus = initial_params + epsilon
        paramsMinus = initial_params - epsilon
        
        costPlus = costFuncRec(paramsPlus, Ytmp, Rtmp, num_of_users, 
                               num_of_movies, num_of_features, lamda)
        
        costMinus = costFuncRec(paramsMinus, Ytmp, Rtmp, num_of_users, 
                               num_of_movies, num_of_features, lamda)
        
        numGrad = (costPlus - costMinus)/(2*eps)
        
        print('Numerical grad:',numGrad,' Expected grad: ',gradParams[rnum])

initial_params = np.hstack((Xtmp.flatten(), Thetatmp.flatten()))
gradParams = gradientsRec(initial_params, Ytmp, Rtmp, num_of_users, 
                          num_of_movies, num_of_features, lamda)

print('Checking the gradient...')

checkGradientsRec(initial_params, gradParams, Ytmp, Rtmp, 
                  num_of_users, num_of_movies, num_of_features, lamda)

# =============== New user ratings ======================

my_ratings = np.zeros((np.size(Y, 0), 1))
my_ratings[10]= 4
my_ratings[11]= 4
my_ratings[63] = 5
my_ratings[81]= 3
my_ratings[126] = 5
my_ratings[186]= 5
my_ratings[195]= 4
my_ratings[199] = 5
my_ratings[203] = 4.5
my_ratings[257] = 5
my_ratings[256]= 4
my_ratings[271]= 4.5
my_ratings[256]= 4
my_ratings[754]= 4.5

# learning movie ratings

print('\n\nFetching your recommended movies...\n')

Y = np.hstack((my_ratings, Y))
R = np.hstack(((my_ratings != 0), R))

def normalize(Y, R):
    Ymulti = np.multiply(Y, R)
    nonZero = np.count_nonzero(Ymulti, 1)
    Ymean = np.array([np.sum(Ymulti, 1)/nonZero]).T
    Ynorm = Y - Ymean
    return Ynorm, Ymean

Ynorm, Ymean = (normalize(Y, R))
num_of_users = np.size(Y, 1)
num_of_movies = np.size(Y, 0)
num_of_features = 10
lamda = 10
learningRate = 0.001
num_of_iter = 1000

X = np.random.rand(num_of_movies, num_of_features)
Theta = np.random.rand(num_of_users, num_of_features)
initial_params = np.hstack((X.flatten(), Theta.flatten()))

def gradientDescent(params, Y, R, learningRate, num_of_iter, num_of_users, 
                    num_of_movies, num_of_features, lamda):
    X = np.reshape(params[:num_of_movies*num_of_features],
                   (num_of_movies, num_of_features))
    Theta = np.reshape(params[num_of_movies*num_of_features:],
                       (num_of_users, num_of_features))
    #m = np.size(Y, 0)
    for i in range(num_of_iter):
        X = X - (learningRate)*((np.multiply((X@Theta.T),R) - np.multiply(Y,R))@Theta + lamda*X)
        Theta = Theta - (learningRate)*((np.multiply((X@Theta.T),R) - np.multiply(Y,R)).T@X + lamda*Theta)
        
    return np.hstack((X.flatten(), Theta.flatten()))
    

'''optimumParams = op.fmin_cg(costFuncRec, initial_params, fprime = gradientsRec,
                           args = (Ynorm, R, num_of_users, num_of_movies,
                                   num_of_features, lamda), maxiter = 100)'''
optimumTheta = gradientDescent(initial_params, Ynorm, R, learningRate, num_of_iter,
                               num_of_users, num_of_movies, num_of_features, lamda) 
                               
print('\n\ncost at optimum theta: ',costFuncRec(optimumTheta, Ynorm, R, num_of_users, 
                                         num_of_movies, num_of_features, lamda))
print('\n\nRecommender trained!!!')

Xopt = np.reshape(optimumTheta[:num_of_movies*num_of_features],
                   (num_of_movies, num_of_features))
ThetaOpt = np.reshape(optimumTheta[num_of_movies*num_of_features:],
                       (num_of_users, num_of_features))


# ============== Recommendations for new Users =====================

predVec = Xopt@ThetaOpt.T

def loadMovieList(num_of_movies):
    movieList = []
    f = open('movie_ids.txt','r')
    for i in range(num_of_movies):
        line = f.readline()
        modLine = line.split(' ', 1)
        newModLine = modLine[1].strip('\n')
        movieList.append(newModLine)    
    return movieList

movieList = loadMovieList(num_of_movies)       
    
myRecommendation = predVec[:, 0].flatten() + Ymean.flatten()
myRecSort = np.argsort(myRecommendation)
myRecSort = myRecSort[::-1]
print(myRecSort)
print('\n\nTop 10 movies for you are:\n\n')
for i in range(10):
    print(movieList[myRecSort[i]])






    
    


         
        
    
    

    

        