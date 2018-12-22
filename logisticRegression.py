# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class mLogisticRegression(object):
    def __init__(self,alpha=0.01, maxiter=100, opttype="SGD"):
        self.alpha = alpha
        self.max_iter = maxiter
        self.opt_type = opttype
        self.coef_ = None
        self.intercept_ = None
        self.coef_list = [[1],[1],[1]]

    def _sigmoid(self,in_x):
        return 1.0/(1+np.exp(-in_x))

    def fit(self,X,y):
        starttime = time.time()
        n_sample, n_feature = X.shape
        self.coef_ = np.ones((n_feature,1))

        for k in range(self.max_iter):
            if self.opt_type == "GD":
                output = self._sigmoid(np.dot(X,self.coef_))
                error = y - output
                self.coef_ += self.alpha*np.dot(X.T,error)
                self.coef_list[0].append(self.coef_[0][0])
                self.coef_list[1].append(self.coef_[1][0])
                self.coef_list[2].append(self.coef_[2][0])
            elif self.opt_type == "SGD":
                for i in range(n_sample):
                    output = self._sigmoid(np.dot(X[i,:].reshape(1,-1),self.coef_))
                    error = y[i] - output
                    #print(self.coef_)
                    self.coef_ += self.alpha*np.dot(X[i,:].reshape(1,-1).T,error)
                    self.coef_list[0].append(self.coef_[0][0])
                    self.coef_list[1].append(self.coef_[1][0])
                    self.coef_list[2].append(self.coef_[2][0])
            elif self.opt_type == "XSGD":
                for i in range(n_sample):
                    nidx = int(np.random.random()*n_sample)
                    self.alpha = 4.0/(1+k+i)+0.01
                    output = self._sigmoid(np.dot(X[nidx,:].reshape(1,-1),self.coef_))
                    error = y[nidx] - output
                    self.coef_ += self.alpha*np.dot(X[nidx,:].reshape(1,-1).T,error)
                    self.coef_list[0].append(self.coef_[0][0])
                    self.coef_list[1].append(self.coef_[1][0])
                    self.coef_list[2].append(self.coef_[2][0])
            else:
                raise NameError("Unsupport optimize method")
        endtime = time.time()
        print("Train process cost {} sec".format(endtime-starttime))


    def predict(self,X):
        n_sample,n_feature = X.shape
        assert n_feature == len(self.coef_)
        labels = []
        for i in range(n_sample):
            label = self._sigmoid(np.dot(X[i,:].reshape(1,-1),self.coef_))[0,0] > 0.5
            if label:
                labels.append(1)
            else:
                labels.append(0)
        return labels

    def score(self,X):
        pass 

def load_data(file):
    X_data = []
    y_data = []
    with open(file,'r') as fin:
        for line in fin:
            x1,x2,y =line.strip().split('\t')
            X_data.append([1.0,float(x1),float(x2)])
            y_data.append([int(y)])

    return np.array(X_data),np.array(y_data)

def showLogRegress(weights, train_x, train_y):
    num_samples, num_features = train_x.shape
    if num_features !=3:
        print("cannt draw")
        return 1

    for i in range(num_samples):
        if int(train_y[i]) == 0:
            plt.plot(train_x[i,1],train_x[i,2],'or')
        elif int(train_y[i]) == 1:
            plt.plot(train_x[i,1],train_x[i,2],'ob')
    min_x = min(train_x[:,1])
    max_x = max(train_x[:,1])
    print("x_min:{0},x_max:{1}".format(min_x,max_x))
    y_min_x =(-weights[0]-weights[1]*min_x)/weights[2]
    y_max_x =(-weights[0]-weights[1]*max_x)/weights[2]
    print("y_min:{0},y_max:{1}".format(y_min_x,y_max_x))
    plt.plot([min_x,max_x],[y_min_x,y_max_x],'-g')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def showCoef(coef_list):
    fig, axes = plt.subplots(3,1)
    for i,ax in enumerate(axes.flat):
        data = coef_list[i]
        ax.plot(data)
    plt.show()
        
if __name__ == "__main__":
    X, y = load_data('data_set')
    #print(y.shape)
    model = mLogisticRegression(alpha=0.001,maxiter=200,opttype="XSGD")
    model.fit(X,y)
    y_pred = model.predict(X)
    y_pred_array = np.array(y_pred).reshape(-1,1)
    corr_n = 0
    for i in range(y.shape[0]):
        if y[i] == y_pred_array[i]:
            corr_n += 1

    #sk_lr = LogisticRegression(max_iter=100,fit_intercept=False,solver="sag")
    #sk_lr.fit(X,y)
    print("accurancy:{}".format(corr_n/y.shape[0]))
    print(model.coef_)

    #sk_pred = sk_lr.predict(X)
    #print("sk_accu:{}".format(accuracy_score(y,sk_pred)))
    #print(sk_lr.coef_)
    #print(sk_lr.coef_.T)
    #print(sk_lr.intercept_)
    showCoef(model.coef_list)
    showLogRegress(model.coef_,X,y)
    #showLogRegress(sk_lr.coef_.T,X,y)
