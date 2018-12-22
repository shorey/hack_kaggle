# coding=utf-8
import numpy as np

class decisiontree(object):
    def __init__(self,maxdepth=1,minsize=10,criterion="gini"):
        self.max_depth = maxdepth
        self.min_size = minsize
        self.criterion = criterion
        self.root = None


    def fit(self,dataset):
        #dataset = np.concatenate((X,y),axis=1)
        root = self._get_split(dataset)
        self._split(root,1)
        self.root = root

    def predict(self,X):
        pass

    def _split(self, node, depth):
        '''create child splits for a node or make terminal'''
        left,right = node['left'],node['right']
        del node['left']
        del node['right']
        if len(left) == 0:
            node['left'] = node['right'] = self._to_terminal(right)
            return
        if len(right)==0:
            node['left'] = node['right'] = self._to_terminal(left)
            return

        if depth >= self.max_depth:
            node['left'],node['right'] = self._to_terminal(left),self._to_terminal(right)
            return

        if len(left) > self.min_size:
            node['left'] = self._get_split(left)
            self._split(node['left'],depth+1)
        else:
            node['left'] = self._to_terminal(left)

        if len(right) > self.min_size:
            node['right'] = self._get_split(right)
            self._split(node['right'],depth+1)
        else:
            node['right'] = self._to_terminal(right)


    def _get_split(self, dataset):
        '''get the best split of a dataset'''
        n_samples, n_features = len(dataset),len(dataset[0])
        classes = set([d[-1] for d in dataset])
        n_features -= 1
        min_score = 999
        node = {'left':[],'right':[],'index':None,'value':None,'score':-1}
        for f_idx in range(n_features):
            f_values = set([d[f_idx] for d in dataset])
            for value in f_values:
                left, right = self._groups_split(f_idx,value,dataset)
                if self.criterion == "gini":
                    score = self._gini_score(left, right, classes)
                    if score < min_score:
                        min_score = score
                        node['left'] = left
                        node['right'] = right
                        node['index'] = f_idx
                        node['value'] = value
                        node['score'] = min_score
                else:
                    raise("TODO")
        return node
        
    
    def _groups_split(self,f_index,value,dataset):
        '''split the dataset by certain feature and certain value,
           return two parts indexs
        '''
        left_idxs = []
        right_idxs = []
        for i in range(len(dataset)):
            if dataset[i][f_index] < value:
                left_idxs.append(dataset[i])
            else:
                right_idxs.append(dataset[i])
        return left_idxs,right_idxs


    def _to_terminal(self,leaf_data):
        y = [row[-1] for row in leaf_data]
        max_cnt = 0
        val = -1
        for i in set(y):
            if y.count(i) > max_cnt:
                max_cnt = y.count(i)
                val = i
        return val 

    def _gini_score(self,left,right,classes):
        left_size = len(left)
        right_size = len(right)
        left_gini = 0 
        right_gini = 0 
        for c in classes:
            if left_size !=0:
                left_gini += ([row[-1] for row in left].count(c)/left_size)**2
            if right_size != 0:
                right_gini += ([row[-1] for row in right].count(c)/right_size)**2
        gini = (1-left_gini)*left_size/(left_size+right_size)+(1-right_gini)*right_size/(left_size+right_size)
        return gini

    def _entropy_gain(self,groups,classes):
        pass 


def load_data(file):
    dataset = []
    with open(file,'r') as fin:
        for line in fin:
            dataset.append([float(x) for x in line.strip().split(',')])
    return dataset

if __name__ == "__main__":
    X = load_data("data_banknote_authentication.txt")
    model = decisiontree(maxdepth=5)
    model.fit(X)
    print(model.root)
