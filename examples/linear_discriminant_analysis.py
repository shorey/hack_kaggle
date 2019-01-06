from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

from supervised_learning import LDA
from utils import calculate_correlation_matrix,accuracy_score
from utils import normalize, standardize, train_test_split, Plot
from unsupervised_learning import PCA

def main():
    #load the dataset
    data = datasets.load_iris()
    X = data.data
    y = data.target

    #three -> two classes
    X = X[y != 2]
    y = y[y != 2]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    #fit and predict using LDA
    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy:",accuracy)
    Plot().plot_in_2d(X_test, y_pred,title="LDA", accuracy=accuracy)

if __name__ == "__main__":
    main()