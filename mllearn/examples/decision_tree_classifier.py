import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import sys
import os

from utils import train_test_split, standardize, accuracy_score
from utils import mean_squared_error,calculate_variance, Plot
from supervised_learning import ClassificationTree

def main():
    print("-- classification Tree --")
    data = datasets.load_iris()
    X = data.data
    y = data.target
    import pdb
    #pdb.set_trace()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = ClassificationTree(min_samples_split=20,min_impurity=0.5)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:",accuracy)

    Plot().plot_in_2d(X_test, y_pred, title="Decision Tree", accuracy=accuracy, legend_labels=data.target_names)


if __name__ == "__main__":
    main()