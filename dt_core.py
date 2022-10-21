# version 1.1
import math
from typing import List
from anytree import *
import sys
from copy import deepcopy


import dt_global

from dt_provided import *
from datetime import datetime


# mylist = read_data("data.csv")

# feature_names = ["mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc"]

# input_feature_names = dt_global.feature_names[:-1]

# data = [[5.2,3.5,1.2,0.1,0], [5.2,3.5,1.2,0.1,0], [5.0,3.5,1.8,0.2,1], [5.3,3.5,1.8,0.2,1], [5.5,3.5,1.8,0.2,0]]


def valid_split(examples: List, feature: str, first_index: int, second_index: int):
    """
    return whether the split between index1 and index2 is a valid split
    data: List[List[Any]]
    """

    feature_index = dt_global.feature_names.index(feature)
    label_index = dt_global.label_index

    # total number of examples
    num = len(examples)
    X = examples[first_index][feature_index]
    Y = examples[second_index][feature_index]
    # we don't want X and Y to be the same 
    if math.isclose(X, Y, rel_tol = 1e-5): return False
    # all the labels from data where the feature takes the value X/Y
    # Y should be greater than x
    Lx = []
    Ly = []
    i1 = 0
    while first_index - i1 >= 0:
        if math.isclose(examples[first_index - i1][feature_index], X, rel_tol = 1e-5):
            Lx.append(examples[first_index - i1][label_index])
        else:
            break
        i1 += 1

    i2 = 0
    while second_index + i2 < num:
        if math.isclose(examples[second_index + i2][feature_index], Y, rel_tol = 1e-5):
            Ly.append(examples[second_index + i2][label_index])
        else:
            break
        i2 += 1

    for i in Lx:
        for j in Ly:
            if i != j: return True
    return False


def get_splits(examples: List, feature: str) -> List[float]:
    """
    Given some examples and a feature, returns a list of potential split point values for the feature.
    
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :return: a list of potential split point values 
    :rtype: List[float]
    """
    
    
    # total number of data
    num = len(examples)

    # the index of the target feature among all features
    feature_index = dt_global.feature_names.index(feature)

    # first sort the data
    
    # myexamples = list(map(list, examples))

    examples.sort(key=lambda x: x[feature_index])

    # the potential split points
    split = []

    for i in range(0, num - 1):
        if valid_split(examples, feature, i, i + 1):
            x = examples[i][feature_index]
            y = examples[i + 1][feature_index]
            temp = (x + y) / 2

            split.append(temp)

    return split


def entropy(d):
    """
    return the entropy of values of d,
    where d is a dictionary
    :parameter type: dictionary {float, int}
    """
    temp = list(d.items())
    # l is a list containing all values in d
    l = []
    for i in temp:
        l.append(i[1])
    sum = 0
    for i in l:
        sum += i * math.log2(i)
    sum = 0 - sum
    return sum

   


def split_entropy(examples: List, feature: str, split: float):
    """
    find the expected entropy after tesing the feature with respect to split
    :rtype: float
    """
    total_size = len(examples)

    Iafter = 0

    leq_examples, gt_examples = split_examples(examples, feature, split)
    
    leq_size = len(leq_examples)
    gt_size = len(gt_examples)

    # leq_labels is a dictionary where the key is the label itself,
    # corresponding value is the number of the occurances of that label
    leq_labels = {}
    for i in leq_examples:
        cur_label = i[dt_global.label_index]
        # label already exists
        if cur_label in leq_labels:
            leq_labels[cur_label] += 1 / leq_size
        else:
            leq_labels[cur_label] = 1 / leq_size

    Ileq = entropy(leq_labels)

    gt_labels = {}
    for i in gt_examples:
        cur_label = i[dt_global.label_index]
        # label already exists
        if cur_label in gt_labels:
            gt_labels[cur_label] += 1 / gt_size
        else:
            gt_labels[cur_label] = 1 / gt_size
    Igt = entropy(gt_labels)    
    
    Iafter = (leq_size / total_size) * Ileq + (gt_size / total_size) * Igt

    return Iafter


def best_split(examples: List, feature: str):
    """
    given a feature, determine the split point based on the info gain
    return the best split point and the corresponding information gain
    
    Tie breaking rules:
    (1) With multiple split points, choose the one with the smallest value.
    
    :type examples: List[List[Any]]
    :rtype: (float, float)
    """
    # total number of data
    num = len(examples)

    # first calculate Ibefore

    # labels is a dictionary where the key is the label itself,
    # corresponding value is the number of the occurances of that label
    labels = {}
    for i in examples:
        # label already exists
        cur_label = i[dt_global.label_index]
        if cur_label in labels:
            labels[cur_label] += 1 / num
        else:
            labels[cur_label] = 1 / num
    Ibefore = entropy(labels)

    # calculate Iafter for each valid split point
    # also use a dictionary storing the info gain of each split point
    valid_splits = get_splits(examples, feature)

    if valid_splits == []: return None, None
    
    max_infogain = (-1) * math.inf
    min_split = math.inf

    for i in valid_splits:
        cur_infogain = Ibefore - split_entropy(examples, feature, i)
        if cur_infogain - max_infogain > 1e-5:
            min_split = i
            max_infogain = cur_infogain            
        elif math.isclose(max_infogain, cur_infogain, rel_tol = 1e-5): # perform tie breaking
            if min_split - i > 1e-5:
                min_split = i          
    
    return min_split, max_infogain


def choose_feature_split(examples: List, features: List[str]) -> (str, float):
    """
    Given some examples and some features,
    returns a feature and a split point value with the max expected information gain.

    If there are no valid split points for the remaining features, return None and -1.

    Tie breaking rules:
    (1) With multiple split points, choose the one with the smallest value. 
    (2) With multiple features with the same info gain, choose the first feature in the list.

    :param examples: a set of examples
    :type examples: List[List[Any]]    
    :param features: a set of features
    :type features: List[str]
    :return: the best feature and the best split value
    :rtype: str, float
    """

    # first for each feature, determine the split point
    # create a dictionary, where the key is the feature, the value is 
    # a tuple (split point, corresponding info gain)

    features_infogain = {}

    # test if there is no split points for any feature,
    # if that is the case, return None and -1
    no_split = 1
    
    
    max_infogain = (-1)*math.inf
    min_index = 0
    for f in features:
        temp = best_split(examples, f)
        
        if temp[0] != None and temp[1] != None:
            no_split = 0
            
            if temp[1] - max_infogain > 1e-5:
                
                max_infogain = temp[1]
                min_index = features.index(f)
            elif math.isclose(temp[1], max_infogain, rel_tol = 1e-5): # perform tie breaking
                
                if features.index(f) - min_index < -1e-5:
                    min_index = features.index(f)
        
    # NO NEED TO PERFROM TIE BREAKING, SINCE IT'S 
    # AUTOMATICALLY SORTED BY THE ORDER I INSERTED

    if no_split == 1:
        return None, -1
    else:
        
        return features[min_index], best_split(examples, features[min_index])[0]


def split_examples(examples: List, feature: str, split: float) -> (List, List):
    """
    Given some examples, a feature, and a split point,
    splits examples into two lists and return the two lists of examples.

    The first list of examples have their feature value <= split point.
    The second list of examples have their feature value > split point.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :param split: the split point
    :type split: float
    :return: two lists of examples split by the feature split
    :rtype: List[List[Any]], List[List[Any]]
    """
    first = []
    second = []
    feature_index = dt_global.feature_names.index(feature)
    
    for i in examples:
        if i[feature_index] - split < -1e-5:
            first.append(i)
        elif math.isclose(i[feature_index], split, rel_tol = 1e-5):
            first.append(i)
        else:
            second.append(i)

    return first, second


def find_decision(examples: List):
    """
    Make a decision based on the examples, use majority vote
    
    Tie breaking: If there is a tie, return the label with the smallest value.
    
    :type examples: List[List[Any]]
    """

    label = {}

    for i in examples:
        cur_label = i[dt_global.label_index]
        # label value already exists
        if cur_label in label:
            label[cur_label] += 1
        else:
            label[cur_label] = 1
    label = dict(sorted(label.items(), key=lambda item: (-item[1], item[0])))

    return list(label.items())[0][0]


def split_node(cur_node: Node, examples: List, features: List[str], max_depth=math.inf):
    """
    Given a tree with cur_node as the root, some examples, some features, and the max depth,
    grows a tree to classify the examples using the features by using binary splits.

    If cur_node is at max_depth, makes cur_node a leaf node with majority decision and return.

    This function is recursive.

    :param cur_node: current node
    :type cur_node: Node
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the maximum depth of the tree
    :type max_depth: int
    """
    
    
    split = cur_node.split
    feature = cur_node.feature

    if cur_node.depth == max_depth:
        cur_node.decision = find_decision(examples)
        cur_node.leaf = True
        
        return

    # find two sets of splitted examples
    splitted = split_examples(examples, feature, split)
    left_example = splitted[0]
    right_example = splitted[1]

    # base on those two sets, grow the children respectively
    feature_left, split_left = choose_feature_split(left_example, features)
    feature_right, split_right = choose_feature_split(right_example, features)

    

    # check whether left is a leaf node
    if feature_left == None:
        child_left = Node('/1', parent=cur_node,
                          feature=None, split=-1, num_examples=len(left_example),
                          depth=cur_node.depth + 1, decision=find_decision(left_example),
                          leaf = True)
    else:
        child_left = Node('/1', parent=cur_node,
                          feature=feature_left, split=split_left,
                          num_examples=len(left_example),
                          depth=cur_node.depth + 1, decision=find_decision(left_example),
                          leaf = False)

        split_node(child_left, left_example, features, max_depth)

    if feature_right == None:

        child_right = Node('/2', parent=cur_node,
                           feature=None, split=-1, num_examples=len(right_example),
                           depth=cur_node.depth + 1, decision=find_decision(right_example),
                           leaf = True)
    else:

        child_right = Node('/2', parent=cur_node,
                           feature=feature_right, split=split_right,
                           num_examples=len(right_example),
                           depth=cur_node.depth + 1, decision=find_decision(right_example),
                           leaf = False)

        split_node(child_right, right_example, features, max_depth)

    return
    
    
    
    
def learn_dt(examples: List, features: List[str], max_depth=math.inf) -> Node:
    """
    Given some examples, some features, and the max depth,
    creates the root of a decision tree, and
    calls split_node to grow the tree to classify the examples using the features, and
    returns the root node.

    This function is a wrapper for split_node.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the max depth of the tree
    :type max_depth: int, default math.inf
    :return: the root of the tree
    :rtype: Node
    """
    feature, split = choose_feature_split(examples, features)
    
    root = Node('root', parent=None, feature=feature,
                num_examples=len(examples),
                split=split, depth=0, decision=find_decision(examples),
                leaf = False)
    split_node(root, examples, features, max_depth)

    return root


def predict(cur_node: Node, example, max_depth=math.inf, \
            min_num_examples=0) -> int:
    """
    Given a tree with cur_node as its root, an example, and optionally a max depth,
    returns a prediction for the example based on the tree.

    If max_depth is provided and we haven't reached a leaf node at the max depth, 
    return the majority decision at this node.

    If min_num_examples is provided and the number of examples at the node is less than min_num_examples, 
    return the majority decision at this node.
    
    This function is recursive.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param cur_node: cur_node of a decision tree
    :type cur_node: Node
    :param example: one example
    :type example: List[Any]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :param min_num_examples: the minimum number of examples at a node
    :type min_num_examples: int, default 0
    :return: the decision for the given example
    :rtype: int
    """
    
    # if cur_node is a leaf, we directly return the decision
    if cur_node.leaf:
        return cur_node.decision
    # check max_depth
    if cur_node.depth == max_depth:
        return cur_node.decision
    # check_min_num_examples
    if cur_node.num_examples < min_num_examples:
        return cur_node.decision

    feature_index = dt_global.feature_names.index(cur_node.feature)

    if example[feature_index] - cur_node.split < -1e-5:
        return predict(cur_node.children[0], example, max_depth, min_num_examples)
    elif math.isclose(example[feature_index], cur_node.split, rel_tol = 1e-5):
        return predict(cur_node.children[0], example, max_depth, min_num_examples)
    else:
        return predict(cur_node.children[1], example, max_depth, min_num_examples)  




def get_prediction_accuracy(cur_node: Node, examples: List, max_depth=math.inf, \
                            min_num_examples=0) -> float:
    """
    Given a tree with cur_node as the root, some examples, 
    and optionally the max depth or the min_num_examples, 
    returns the accuracy by predicting the examples using the tree.

    The tree may be pruned by max_depth or min_num_examples.

    :param cur_node: cur_node of the decision tree
    :type cur_node: Node
    :param examples: the set of examples. 
    :type examples: List[List[Any]]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :param min_num_examples: the minimum number of examples at a node
    :type min_num_examples: int, default 0
    :return: the prediction accuracy for the examples based on the cur_node
    :rtype: float
    """
    total = len(examples)
    
    # correct stores the number of prediction success
    correct = 0
    
    for l in examples:
        prediction = predict(cur_node, l, max_depth, min_num_examples)
        real = l[dt_global.label_index]
        if prediction == real:
            correct += 1
    
    
    return correct/total


def post_prune(cur_node: Node, min_num_examples: float):
    """
    Given a tree with cur_node as the root, and the minimum number of examples,
    post prunes the tree using the minimum number of examples criterion.

    This function is recursive.

    Let leaf parents denote all the nodes that only have leaf nodes as its descendants. 
    Go through all the leaf parents.
    If the number of examples at a leaf parent is smaller than the pre-defined value,
    convert the leaf parent into a leaf node.
    Repeat until the number of examples at every leaf parent is greater than
    or equal to the pre-defined value of the minimum number of examples.

    :param cur_node: the current node
    :type cur_node: Node
    :param min_num_examples: the minimum number of examples
    :type min_num_examples: float
    """
    
    # if cur_node is a leaf 
    
    if cur_node.leaf == True: return
    
        
    # if cur_node has only one child
    if cur_node.children[0] == None and cur_node.children[1] != None:
        post_prune(cur_node.children[1], min_num_examples)
    elif cur_node.children[1] == None and cur_node.children[0] != None:
        post_prune(cur_node.children[0], min_num_examples)
    else: # if cur_node has 2 childs
        post_prune(cur_node.children[0], min_num_examples)
        post_prune(cur_node.children[1], min_num_examples)
        
    # check whether this is a leaf parent
    if cur_node.children[0].leaf and cur_node.children[1].leaf:
        if cur_node.num_examples < min_num_examples:
            left_child = cur_node.children[0]
            right_child = cur_node.children[1]
            left_child.parent = None
            right_child.parent = None
            cur_node.leaf = True
            return
        
def post_prune_check(cur_node: Node, min_num_examples: float):
    if cur_node.leaf == True: return
    if cur_node.children[0].leaf and cur_node.children[1].leaf:
        print(cur_node.num_examples)
        if cur_node.num_examples < min_num_examples: 
            print("FALSE")
            return
            
    
    if cur_node.children[0] == None and cur_node.children[1] != None:
        post_prune_check(cur_node.children[1], min_num_examples)
    elif cur_node.children[1] == None and cur_node.children[0] != None:
        post_prune_check(cur_node.children[0], min_num_examples)
    else: # if cur_node has 2 childs
        post_prune_check(cur_node.children[0], min_num_examples)
        post_prune_check(cur_node.children[1], min_num_examples)
        
    
    
"""
start_time = datetime.now()
sys.stdout = open('tree.txt', 'w')
tree = learn_dt(mylist, feature_names, 5)
end_time = datetime.now()

print(RenderTree(tree))

print('Duration: {}'.format(end_time - start_time))
sys.stdout.close()
"""

"""
input_feature_names = dt_global.feature_names[:-1]
full_tree = learn_dt(test, input_feature_names)
prediction = predict(full_tree, [5.6, 3.3, 2.1, 0.1, 0])
print(prediction)
"""
# tree = learn_dt(data, input_feature_names)
# post_prune(tree, 4)

"""
examples=[[0.37, 0.46, 0.4, 0.27, 0.5, 0.0, 0.45, 0.22, 7], [0.3, 0.37, 0.4, 0.45, 0.5, 0.0, 0.48, 0.41, 5], [0.42, 0.51, 0.39, 0.22, 0.5, 0.0, 0.49, 0.26, 5], [0.39, 0.63, 0.41, 0.27, 0.5, 0.0, 0.49, 0.22, 5], [0.51, 0.46, 0.41, 0.36, 0.5, 0.0, 0.49, 0.22, 5], [0.36, 0.5, 0.39, 0.23, 0.5, 0.0, 0.5, 0.22, 5], [0.51, 0.47, 0.41, 0.44, 0.5, 0.0, 0.5, 0.22, 5], [0.44, 0.52, 0.41, 0.23, 0.5, 0.0, 0.51, 0.22, 7], [0.43, 0.28, 0.41, 0.32, 0.5, 0.0, 0.52, 0.22, 5]]

examples = [[0.48, 0.61, 0.44, 0.2, 0.5, 0.0, 0.47, 0.22, 0], [0.42, 0.71, 0.45, 0.15, 0.5, 0.0, 0.48, 0.22, 5], [0.46, 0.41, 0.45, 0.23, 0.5, 0.0, 0.48, 0.22, 7], [0.46, 0.47, 0.46, 0.16, 0.5, 0.0, 0.51, 0.22, 0], [0.48, 0.49, 0.45, 0.23, 0.5, 0.0, 0.51, 0.22, 5], [0.47, 0.41, 0.46, 0.18, 0.5, 0.0, 0.53, 0.22, 5], [0.46, 0.48, 0.44, 0.21, 0.5, 0.0, 0.55, 0.22, 8], [0.46, 0.49, 0.45, 0.28, 0.5, 0.0, 0.55, 0.22, 9]]

print(choose_feature_split(examples, feature_names))
e1 = split_examples(examples, choose_feature_split(examples, feature_names)[0], choose_feature_split(examples, feature_names)[1])[0]

print(choose_feature_split(e1, feature_names))
"""
"""
tree = learn_dt(mylist, feature_names)

for i in range(0, 301, 20):
    print("new", i)
    post_prune(tree, i)
    post_prune_check(tree, i)
"""