import numpy as np
import math
from utils import train_test_split, normalize
from utils import Plot, accuracy_score

class NaiveBayes():
    """
    The gaussian Naive Bayes classifier
    """
    def fit(self, X, y):
        self.X, self.y = X, y
        self.classes = np.unique(y)
        self.parameters = []
        # calculate the mean and variance of each feature for each class
        for i, c in enumerate(self.classes):
            #only select the rows where the label equals the given class
            X_where_c = X[np.where(y==c)]
            self.parameters.append([])
            #add the mean and variance for each feature(column)
            for col in X_where_c.T:
                parameters = {"mean":col.mean(), "var":col.var()}
                self.parameters[i].append(parameters)


    def _calculate_likelihood(self, mean, var, x):
        """gaussian likelihood of the data x given mean and var"""
        eps = 1e-4
        coeff = 1.0/math.sqrt(2.0*math.pi*var+eps)
        exponent = math.exp(-(math.pow(x-mean,2))/(2*var+eps))
        return coeff*exponent

    def _calculate_prior(self, c):
        """calculate the prior of class c (samples where class == c/total number of samples)"""
        frequency = np.mean(self.y==c)
        return frequency

    def _classify(self, sample):
        """
        classification using bayes rule P(Y|X) = P(X|Y)*P(Y)/P(X),
        or posterior = Likelihood * Prior/Scaling Factor

        P(Y|X) - The posterior is the probability that sample x is of class y given the
                 feature values of x being distributed according to distribution of y and the prior.
        P(X|Y) - Likelihood of data X given class distribution Y.
                 Gaussian distribution (given by _calculate_likelihood)
        P(Y)   - Prior (given by _calculate_prior)
        P(X)   - Scales the posterior to make it a proper probability distribution.
                 This term is ignored in this implementation since it doesn't affect
                 which class distribution the sample is most likely to belong to.

        classifies the sample as the class thata results in the largest P(Y|X) (posterior)
        :param sample:
        :return:
        """
        posteriors = []
        # go through list of classes

        for i, c in enumerate(self.classes):
            # initialize posterior as prior
            posterior = self._calculate_prior(c)
            # naive assumption (independence):
            # P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)
            # posterior is product of prior and likelihoods (ignoriing acaling factor)
            for feature_value, params in zip(sample, self.parameters[i]):
                # likelihood of feature value given distribution of feature values given y
                likelihood = self._calculate_likelihood(params["mean"], params["var"], feature_value)
                posterior *= likelihood
            posteriors.append(posterior)
        #return the class with the largest posterior porbability
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """predict the class labels of the samples in X"""
        y_pred = [self._classify(sample) for sample in X]
        return y_pred