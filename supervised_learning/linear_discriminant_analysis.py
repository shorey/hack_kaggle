import numpy as np
from utils import calculate_covariance_matrix,normalize
from utils import standardize

class LDA():
    """The Linear Discriminant Analysis classifier, also known as Fisher's linear discriminant.
    Can besides from classification also be used to reduce the dimensionaly of the dataset.
    """
    def __init__(self):
        self.w = None

    def transform(self, X, y):
        self.fit(X, y)
        # project data onto vector
        X_transform = X.dot(self.w)
        return X_transform

    def fit(self,X, y):
        #separate data by class
        X1 = X[y==0]
        X2 = X[y==1]

        #calcualte the covariance matrices of the two datasets
        cov1 = calculate_covariance_matrix(X1)
        cov2 = calculate_covariance_matrix(X2)

        cov_tot = cov1 + cov2

        #calculate the mean of the two datasets
        mean1 = X1.mean(0)
        mean2 = X2.mean(0)
        mean_diff = np.atleast_1d(mean1 - mean2)

        #determine the vector which when X is projected
        #onto its best separates the data by class w=(mean1-mean2)/(cov1+cov2)
        self.w = np.linalg.pinv(cov_tot).dot(mean_diff)

    def predict(self, X):
        y_pred = []
        for sample in X:
            h = sample.dot(self.w)
            y = 1*(h < 0)
            y_pred.append(y)
        return y_pred
