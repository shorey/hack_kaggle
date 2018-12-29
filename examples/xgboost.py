import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import progressbar
from utils import train_test_split
from utils import standardize
from utils import to_categorical
from utils import normalize
from utils import mean_squared_error,accuracy_score,Plot
from supervised_learning import XGBoost

def main():
    print("--XGBoost--")
    data = datasets.load_iris()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = XGBoost(n_estimators=200,max_depth=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    print("Accuracy:",accuracy)

    Plot().plot_in_2d(X_test, y_pred,
                      title="XGBoost",
                      accuracy=accuracy,
                      legend_labels=data.target_names)

if __name__ == "__main__":
    main()