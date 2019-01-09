import math
import numpy as np

from utils import train_test_split, to_categorical, normalize, accuracy_score
from deep_learning.activation_functions import Sigmoid, ReLU, SoftPlus, LeakyReLU, TanH, ELU
from deep_learning.loss_functions import CrossEntropy, SquareLoss
from utils import Plot
from utils.misc import bar_widgets
import progressbar

class Perceptron():
    """
    the perceptron. One layer neural network classifier.
    parameter:
    ____________
    n_iterations: float
        the number of training iterations the algorithm will tune the weights for

    activation_function: class
        the activation that shall be used for each neuron
        possible choices: sigmoid, ExpLU, ReLU, LeakyReLU, SoftPlus, TanH

    loss: class
        the loss function used to assess the model's performance.
        possible choices: Squareloss, CrossEntropy

    learning_rate: float
        the step length that will be used when updating the weight.
    """
    def __init__(self, n_iterations=20000,activation_function=Sigmoid, loss=SquareLoss,
                 learning_rate=0.01):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.loss = loss()
        self.activation_func = activation_function()
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)


    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        _, n_outputs = np.shape(y)

        #initialize weights between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features,n_outputs))
        self.w0 = np.zeros((1, n_outputs))

        for i in self.progressbar(range(self.n_iterations)):
            #calculate outputs
            linear_output = X.dot(self.W) + self.w0
            y_pred = self.activation_func(linear_output)
            #calculate the loss gradient w.r.t the input of the activation function
            error_gradient = self.loss.gradient(y, y_pred)*self.activation_func.gradient(linear_output)
            # Calculate the gradient of the loss with respect to each weight
            grad_wrt_w = X.T.dot(error_gradient)
            grad_wrt_w0 = np.sum(error_gradient, axis=0, keepdims=True)
            # update weights
            self.W -= self.learning_rate * grad_wrt_w
            self.w0 -= self.learning_rate * grad_wrt_w0

    # use the trained model to predict labels of X
    def predict(self, X):
        y_pred = self.activation_func(X.dot(self.W) + self.w0)
        return y_pred
