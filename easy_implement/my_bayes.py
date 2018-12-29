# coding=utf-8
import numpy as np

class MMultinomialNB(object):
    def __init__(self, alpha=1.0, class_prob_prior=None):
        self.alpha = alpha
        self.class_prob_prior = class_prob_prior
        self.condition_prob = None
        self.classes = None
        self.feature_num = None
        self.sample_num = None
        self.y = None

    def _calculate_class_prob_prior(self,y):
        self.sample_num = len(y)
        if self.class_prob_prior is None:
            self.class_prob_prior = {}
            for c in self.classes:
                self.class_prob_prior[c] = ((np.sum(y==c)+self.alpha)*1.0/(self.sample_num + len(self.classes)*self.alpha))

    def _calculate_condition_prob(self,feature):
        values = np.unique(feature)
        nums = len(feature)
        prob_dict = {}
        for v in values:
            prob_dict[v] = (np.sum(feature==v)+self.alpha)*1.0/(nums+self.feature_num*self.alpha)
        return prob_dict

    def _get_xj_condition_prob(self,fidx,xj,yi):
        n_yi = np.sum(self.y == yi)
        return self.condition_prob.get(yi).get(fidx).get(xj,self.alpha/(n_yi+self.feature_num*self.alpha))

    def _predict_one_sample(self,X):
        label = None
        max_posteriori_prob = 0.0
        current_posteriori_prob = 1.0
        feature = X
        assert len(feature) == self.feature_num,"input feature col not match"
        for c in self.classes:
            current_prob = 1.0
            for i in range(len(feature)):
                current_prob *= self._get_xj_condition_prob(i,feature[i],c)
            current_posteriori_prob = self.class_prob_prior.get(c)*current_prob 
            if current_posteriori_prob > max_posteriori_prob:
                max_posteriori_prob = current_posteriori_prob
                label = c
        return label


    def fit(self ,X ,y):
        self.classes = np.unique(y)
        self.y = y
        self.feature_num = X.shape[1]
        self._calculate_class_prob_prior(y)

        self.condition_prob = {}
        for c in self.classes:
            self.condition_prob[c] = {}
            for i in range(self.feature_num):
                feature = X[y==c][:,i]
                self.condition_prob[c][i] = self._calculate_condition_prob(feature)
        return self

    def predict(self,X):
        sample_num = X.shape[0]
        labels = []
        for i in range(sample_num):
            feature = X[i]
            label = self._predict_one_sample(feature)
            labels.append(label)
        return labels

if __name__ == "__main__":
    X = np.array([
                      [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
                      [4,5,5,4,4,4,5,5,6,6,6,5,5,6,6]
             ])
    X = X.T
    t_x = np.array([[1,4],[2,5],[1,6],[0,7]])
    y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    NB = MMultinomialNB(alpha=1.0)
    #NB = MMultinomialNB()
    NB.fit(X,y)
    #print(NB.predict(np.array([[3,5]])))
    print(NB.predict(X))
    
