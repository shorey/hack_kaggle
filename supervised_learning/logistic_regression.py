import numpy as np
import progressbar
from utils.misc import bar_widgets
import math
from utils import make_diagonal, Plot, shuffle_data
from deep_learning.activation_functions import Sigmoid


class LogisticRegression():
    """
    logistic regression classifier.

    parameters:
    ------------
    learning_rate: float
        the step length that will be taken when following the
        negative gradient during training

    gradient_descent:boolean
        true or false depending if gradient descent should be used
        when training. if false then we use batch optimization by least squares.

    """
    def __init__(self, learning_rate=0.1, gradient_descent=True, opt_type="GD"):
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.opt_type = opt_type
        self.n_features = None
        self.n_samples = None
        self.sigmoid = Sigmoid()
        self.bar = progressbar.ProgressBar(widgets=bar_widgets)


    def _initialize_parameters(self, X):
        self.n_sample,self.n_features = np.shape(X)
        # initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1/math.sqrt(self.n_features)
        self.param = np.random.uniform(-limit, limit, (self.n_features,))


    def fit(self, X, y, n_iterations=4000):
        self._initialize_parameters(X)
        #tune parameters for n iterations
        for i in self.bar(range(n_iterations)):
            if self.gradient_descent:
                #move against the gradient of the loss function with
                #respect to the parameters to minimize the loss
                if self.opt_type == "GD":
                    y_pred = self.sigmoid(X.dot(self.param))
                    self.param -= self.learning_rate*-(y-y_pred).dot(X)
                elif self.opt_type == "SGD":
                    X_shuffle, y_shuffle = shuffle_data(X, y)
                    for i in range(self.n_sample):
                        y_pred = self.sigmoid(X_shuffle[i,:].dot(self.param))
                        self.param -= self.learning_rate*-(y_shuffle[i]-y_pred)*X_shuffle[i,:]

            else:
                y_pred = self.sigmoid(X.dot(self.param))
                #make a diagonal matrix of the sigmoid gradient column vector
                diag_gradient = make_diagonal(self.sigmoid.gradient(X.dot(self.param)))
                #batch opt
                self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(
                    diag_gradient.dot(X).dot(self.param) + y - y_pred)

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred
