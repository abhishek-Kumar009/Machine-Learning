from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# loading and visualizing the data

dataset = load_breast_cancer()
feature = dataset['data']
feature_label = dataset['feature_names']
labels = dataset['target']
labels_names = dataset['target_names']

# ========================================

# splitting the dataset into training and test set

X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size = 0.33, random_state = 42)

# ========================================

# building the algorithm model

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# ========================================

# Making the prediction

prediction = gnb.predict(X_test)
print(prediction)
# ========================================

# Finding the accuracy

print("Accuracy : {:.3f}\n".format(gnb.score(X_test, y_test)))

# ========================================


