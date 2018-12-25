from sklearn import linear_model
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import pdb

# =============== dataset for the project =====================================

dataset = fetch_california_housing()


# ============== Polynomial Features for the dataset ==========================

pft = PolynomialFeatures(degree = 3)

# ============== Label names ==================================================

label_prices = dataset['target']
feature_names = dataset['feature_names']

# ============== Feature Normalization of the dataset =========================
data_original = (dataset.data)
X_scaled = preprocessing.scale(dataset.data)

# ================= Generating poly features ==================================

X_poly = pft.fit_transform(X_scaled)

# ================= Splitting the dataset(train, validation and test ==========
X_train, X_dummy, y_train, y_dummy = train_test_split(X_poly, dataset.target, test_size = 0.40, random_state = 42)
X_CV,X_test,y_CV,y_test = train_test_split(X_dummy, y_dummy, test_size = 0.2, random_state = 42)

# ================= Fit a linear regression model =============================
model = linear_model.Ridge(alpha = 9000)
model.fit(X_train, y_train)

predictionCV = model.predict(X_CV)
predictionTestSet = model.predict(X_test)

errorCV = mean_squared_error(y_CV, predictionCV)
errorTestSet = mean_squared_error(y_test, predictionTestSet)


# ================= Plotting graph ============================================

plt.scatter(y_CV, predictionCV, c = 'green')
plt.xlabel("Price in 1000$")
plt.ylabel("Predicted CV value")
plt.title("Predicted CV value vs True CV value: Linear Regression")
plt.show()


print("Predicted Value[60] from test set: {}\n".format(predictionTestSet[60]))
print("Original Value[60] form test set: {}\n".format(y_test[60]))
print("Prices for the houses: {}\n".format(label_prices))
print("Feature names: {}\n".format(feature_names))
print("Prediction in CV: {}\n".format(predictionCV))
print("Original Values of CV: {}\n".format(y_CV))

print("Error in cross-validation set: {:.2f}\n".format(errorCV))

print("Predicted value for test set: {}\n".format(predictionTestSet))
print("Original value for test set: {}\n".format(y_test))

print("Error in test set: {:.2f}\n".format(errorTestSet))


