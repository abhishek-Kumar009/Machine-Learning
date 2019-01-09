import predictLR as pr
import errorCalculate as ec
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import normalization as nm
import gradientDescentLR as gd
import numpy as np
from sklearn.datasets import fetch_california_housing
dataset = fetch_california_housing()
tdf = pd.DataFrame(dataset.data)
tdfr = pd.DataFrame(dataset.target)
Xt = tdf.head(1000)
yt = tdfr.head(1000)
Xts = tdf.tail(500)
yts = tdfr.tail(500)

ytrain = np.array(dataset.data)
Xtrain = np.array(dataset.data)

Xtest = np.array(dataset.data)
ytest = np.array(dataset.data)

XtrainNorm = nm.featureNorm(Xtrain)
XtestNorm = nm.featureNorm(Xtest)

pft = PolynomialFeatures(4, interaction_only = True)

XtrainPoly = pft.fit_transform(XtrainNorm)
XtestPoly = pft.fit_transform(XtestNorm)

initialTheta = np.zeros((XtrainPoly.shape[1] + 1,1))
thetaBest = gd.gradDesc(XtrainPoly, ytrain, initialTheta, 0.01, 2000)
prediction = pr.predict(XtestPoly, thetaBest)
error = ec.errorCalc(prediction, ytest)
print('Accuracy: ',(1-error)*100)






 