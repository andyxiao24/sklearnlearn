import numpy as np

# datasets for training, skipping the trouble to construct the training data yourself
from sklearn import datasets
# load k-neighbors algorithm from sklearn
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

# iris data set
iris = datasets.load_iris()
test = datasets.load_boston()


iris_x = iris.data
iris_y = iris.target

# split the data set into two parts: training set and testing set
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.05)

knn = KNeighborsClassifier()
# train the model
knn.fit(x_train, y_train)

print iris_x
print iris_y

print x_train
print y_train

# use the model
print knn.predict(x_test)

# compare with the real result
print y_test