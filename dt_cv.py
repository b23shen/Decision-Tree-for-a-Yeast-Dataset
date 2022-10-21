# version 1.1
from typing import List

import dt_global
from dt_core import *


def cv_pre_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for pre-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """  
    # get all the features (without label)
    features = []
    num_total_features = len(dt_global.feature_names)
    
    for i in range(num_total_features):
        if i != dt_global.label_index:
            features.append(dt_global.feature_names[i])
            
    # num_value is the number of values we need to consider
    num_value = len(value_list)
    
    # num_folds is the number of tests we need to perform to calculate
    # the avg prediction accuracy for one value
    num_folds = len(folds)
    
    # testing is a list of tuples,
    # each tuple represents: (training examples, validation examples)
    # there are num_folds tuples in list testing
    # for each value, we should consider all tuples in testing
    # training examples type: List[List[float]]
    # validation examples type: List[List[float]]
    testing = []
    for i in range(num_folds):
        validation = folds[i]
        train = []
        for j in range(num_folds):
            if j != i:
                train = [*train, *folds[j]]
                
        testing.append((train, validation))  
    
    
    training_accuracy = [0] * num_value
    validation_accuracy = [0] * num_value 
    
    # an element in training/validation_accuracy is 
    # the average among testing through all examples in testing
    
    for t in testing:
        tree = learn_dt(t[0], features)
        for i in range(num_value):
            
            cur_value = value_list[i]
            
            training_accuracy[i] += get_prediction_accuracy(tree, t[0], max_depth = cur_value)
            validation_accuracy[i] += get_prediction_accuracy(tree, t[1], max_depth = cur_value)
            
    for i in range(num_value):
        training_accuracy[i] = training_accuracy[i]/num_folds
        validation_accuracy[i] = validation_accuracy[i]/num_folds
            
    """
    max_prediction_accuracy = max(validation_accuracy)
    print(max_prediction_accuracy)
    best_max_depth = validation_accuracy.index(max_prediction_accuracy)
    print(best_max_depth)
    l = range(0, 31)
    print(l[best_max_depth])
    """
    return training_accuracy, validation_accuracy

def cv_post_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for post-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """ 
    
    # get all the features (without label)
    features = []
    num_total_features = len(dt_global.feature_names)
    
    for i in range(num_total_features):
        if i != dt_global.label_index:
            features.append(dt_global.feature_names[i])
            
    # num_value is the number of values we need to consider
    num_value = len(value_list)
    
    # num_folds is the number of tests we need to perform to calculate
    # the avg prediction accuracy for one value
    num_folds = len(folds)
    
    # testing is a list of tuples,
    # each tuple represents: (training examples, validation examples)
    # there are num_folds tuples in the list testing
    # for each value, we should consider all tuples in testing
    
    # training examples type: List[List[float]]
    # validation examples type: List[List[float]]
    testing = []
    for i in range(num_folds):
        validation = folds[i]
        train = []
        for j in range(num_folds):
            if j != i:
                train = [*train, *folds[j]]
                
        testing.append((train, validation))  
    
    
    training_accuracy = [0] * num_value
    validation_accuracy = [0] * num_value 
    
    # an element in training/validation_accuracy is 
    # the average among testing through all examples in testing
    
    for t in testing:
        tree = learn_dt(t[0], features)
        for i in range(num_value):
            cur_value = value_list[i]
            post_prune(tree, cur_value)
            training_accuracy[i] += get_prediction_accuracy(tree, t[0])
            validation_accuracy[i] += get_prediction_accuracy(tree, t[1])
            
    for i in range(num_value):
        training_accuracy[i] = training_accuracy[i]/num_folds
        validation_accuracy[i] = validation_accuracy[i]/num_folds
    
    """
    max_prediction_accuracy = max(validation_accuracy)
    print(max_prediction_accuracy)
    best_min_num_examples = validation_accuracy.index(max_prediction_accuracy)
    print(best_min_num_examples)
    
    l = range(0, 301, 20)
    print(l[best_min_num_examples])
    """
    
    return training_accuracy, validation_accuracy



def mean(l: List):
    sum = 0
    for i in l:
        sum += i
    return sum/len(l)

"""
start_time = datetime.now()
print(cv_post_prune(preprocess(mylist), range(0,301,20)))
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

start_time = datetime.now()
print(cv_pre_prune(preprocess(mylist), range(0,31)))
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
"""
"""
start_time = datetime.now()
print(learn_dt(mylist, feature_names))
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
"""