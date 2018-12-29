# coding=utf-8
import numpy as np

class MultinomialNB(object):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes = None
        self.conditional_prob = None
        self.train_num = None

    def _calculate_feature_prob(self, feature):
        values = np.unique(feature)
        total_num = float(len(feature))
        value_prob = {}
        for v in values:
            value_prob[v] = ((np.sum(np.equal(feature,v))+self.alpha)/(total_num + len(values)*self.alpha))
        return value_prob 

    def fit(self,X,y):
        self.train_num = X.shape[1]
        self.classes = np.unique(y)
        if self.class_prior == None:
            class_num = len(self.classes)
            if not self.fit_prior:
                self.class_prior = [1.0/class_num for _ in range(class_num)]
            else:
                self.class_prior = []
                sample_num = float(len(y))
                for c in self.classes:
                    c_num = np.sum(np.equal(y,c))
                    self.class_prior.append((c_num+self.alpha)/(sample_num+class_num*self.alpha))

        self.conditional_prob = {}
        for c in self.classes:
            self.conditional_prob[c] = {}
            for i in range(X.shape[1]):
                feature = X[np.equal(y,c)][:,i]
                self.conditional_prob[c][i] = self._calculate_feature_prob(feature)
        return self

    def _get_xj_prob(self, values_prob, target_value):
        return values_prob.get(target_value, 1.0/self.train_num)

    def _predict_single_sample(self, x):
        label = -1
        max_posterior_prob = 0
        
        for c_index in range(len(self.classes)):
            current_class_prior = self.class_prior[c_index]
            current_conditional_prob = 1.0
            feature_prob = self.conditional_prob[self.classes[c_index]]
            j = 0
            for feature_i in feature_prob.keys():
                current_conditional_prob *= self._get_xj_prob(feature_prob[feature_i],x[j])
                j += 1

            if current_class_prior * current_conditional_prob > max_posterior_prob:
                max_posterior_prob = current_class_prior * current_conditional_prob
                label = self.classes[c_index]
        return label 

    def predict(self,X):
        labels = []
        for i in range(X.shape[0]):
            label = self._predict_single_sample(X[i])
            labels.append(label)
        return labels 

if __name__ == "__main__":
    X = np.array([ [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
                   [4,5,5,4,4,4,5,5,6,6,6,5,5,6,6]])
    X = X.T
    y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    nb = MultinomialNB(alpha=1.0,fit_prior=True)
    nb.fit(X,y)
    print(nb.predict(X))#输出-1
