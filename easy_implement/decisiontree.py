# coding=utf-8
import copy
import numpy as np
from random import randrange
from random import seed
from my_decisiontree import decisiontree

def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right 

def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0 
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val)/size
            score += p*p
        gini += (1 - score)*(size/n_instances)
    return gini 

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value,b_score,b_groups = 999,999,999,None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index,row[index],gini,groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])

    if not left or not right:
        node['left'] = node['right'] = to_terminal(left+right)
        return
    if depth >= max_depth:
        node['left'],node['right'] = to_terminal(left),to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'],max_depth, min_size,depth+1)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'],max_depth,min_size,depth+1)

def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size,1)
    return root

def print_tree(node, depth=0):
    if isinstance(node, dict):
        print("%s[X%s < %.3f]"%((depth*' ',(node['index']+1),node['value'])))
        print_tree(node['left'],depth+1)
        print_tree(node['right'],depth+1)
    else:
        print('%s[%s]'%((depth*' ',node)))

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'],dict):
            return predict(node['left'],row)
        else:
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict(node['right'],row)
        else:
            return node['right']


def load_data(file):
    dataset = []
    with open(file,'r') as fin:
        for line in fin:
            dataset.append([float(x) for x in line.strip().split(',')])
    return dataset

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = copy.deepcopy(dataset)
    fold_size = int(len(dataset)/n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct/float(len(actual))*100

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    for fold in folds:
        tmp_set = copy.deepcopy(folds)
        tmp_set.remove(fold)
        train_set = []
        for tt in tmp_set:
            train_set.extend(tt)

        test_set = fold 
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in test_set]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train,max_depth,min_size)
    predictions = []
    #print(tree)
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions 

def my_decision_tree(train, test, max_depth, min_size):
    model = decisiontree(maxdepth=max_depth, minsize=min_size)
    model.fit(train)
    tree = model.root
    print(tree)
    predictions = []
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions

seed(2)
dataset = []
#dataset = load_data('data_banknote_authentication.txt')
from sklearn import datasets 
data = datasets.load_iris()
X = data.data.tolist()
y = data.target.tolist()

for i in range(len(y)):
    dataset.append(X[i]+[y[i]])

n_folds = 5
max_depth = 5
min_size = 10 
scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth,min_size)
print("scores:%s"%scores)
print("Mean Accuracy:%.3f"%(sum(scores)/float(len(scores))))

scores = evaluate_algorithm(dataset, my_decision_tree, n_folds, max_depth,min_size)
print("scores:%s"%scores)
print("Mean Accuracy:%.3f"%(sum(scores)/float(len(scores))))


