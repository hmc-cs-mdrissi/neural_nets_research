import torch
from torch.autograd import Variable

from six import string_types

from functools import partial
import itertools

class Node:
    """
    Node class
    """
    def __init__(self, value):
        self.value = value
        self.children = []

    def cuda(self):
        return map_tree(lambda value: value.cuda(), self)

def make_tree(json, long_base_case=True):
    # First base case - variable name
    if isinstance(json, string_types):
        if long_base_case:
            parentNode = Node("<VAR>")
            childNode = Node(json)
            parentNode.children.append(childNode)
        else:
            parentNode = Node(json)
        return parentNode

    # Second base case - variable value
    if type(json) is int:
        return Node(json)

    tag = "<" + json["tag"].upper() + ">"
    children = json["contents"]
    
    if not long_base_case:
        if tag == '<CONST>':
            return Node(children)
        
    parentNode = Node(tag)

    if type(children) is list:
        parentNode.children.extend(map(make_tree, children))
    else:
        parentNode.children.append(make_tree(children))

    return parentNode

def binarize_tree(tree):
    new_tree = Node(tree.value)
    curr_node = new_tree

    for child in tree.children:
        new_node = binarize_tree(child)
        curr_node.children.append(new_node)
        curr_node = new_node

    return new_tree

def vectorize(val, num_vars, num_ints, ops, eos_token=False, one_hot=True):
    if type(val) is int:
        index = val
    elif val not in ops:
        index = int(val[1:]) - 1 + num_ints
    else:
        index = num_ints + num_vars + ops[val]

    if one_hot:
        eos_bonus = 1 if eos_token else 0
        return make_one_hot(num_vars + num_ints + len(ops.keys()) + eos_bonus, index)

    return index

def make_one_hot(len, index):
    vector = torch.zeros(len)
    vector[index] = 1
    return Variable(vector)

def un_one_hot(vector):
    return int(vector.data.nonzero())

def map_tree(func, tree):
    new_tree = Node(func(tree.value))
    new_tree.children.extend(map(partial(map_tree, func), tree.children))
    return new_tree

def print_tree(tree):
    print(tree.value)
    for child in tree.children:
        print_tree(child)

def encode_tree(tree, num_vars, num_ints, ops, eos_token=False, one_hot=True):
    return map_tree(lambda node: vectorize(node, num_vars, num_ints, ops, eos_token=eos_token, one_hot=one_hot), tree)

def decode_tokens(seq, num_vars, num_ints, ops):
    reverse_ops = dict(map(lambda p: (p[1], p[0]), ops.items()))

    def index_to_token(index):
        if index < num_ints:
            return index
        elif index < num_ints + num_vars:
            return 'a' + str(index - num_ints + 1)
        else:
            return reverse_ops[index - num_ints - num_vars]

    return list(map(index_to_token, seq))

def tree_to_list(tree):
  """
        Concatenate a tree into a list using a pre-order traversal.

        :param tree: a tree.
        :return a list of values of the tree
  """
  return [tree.value] + list(itertools.chain.from_iterable(map(tree_to_list, tree.children)))

def translate_from_for(tree):
    if tree.value == '<SEQ>':
        t1 = translate_from_for(tree.children[0])
        t2 = translate_from_for(tree.children[1])
        if t1.value == '<LET>' and t1.children[-1].value == '<UNIT>':
            t1.children[-1] = t2
            return t1
        else:
            new_tree = Node('<LET>')
            new_tree.children.extend([Node('a11'), t1, t2])
            return new_tree
    elif tree.value == '<IF>':
        cmp = tree.children[0]
        t1 = translate_from_for(tree.children[1])
        t2 = translate_from_for(tree.children[2])
        new_tree = Node('<IF>')
        new_tree.children.extend([cmp, t1, t2])
        return new_tree
    elif tree.value == '<FOR>':
        var = tree.children[0]
        init = translate_from_for(tree.children[1])
        cmp = translate_from_for(tree.children[2])
        inc = translate_from_for(tree.children[3])
        body = translate_from_for(tree.children[4])

        tb = Node('<LET>')
        tb.children.append(Node('a11'))
        tb.children.append(body)
        
        increment = Node('<APP>')
        increment.children.extend([Node('a12'), inc])
        tb.children.append(increment)

        funcbody = Node('<IF>')
        funcbody.children.extend([cmp, tb, Node('<UNIT>')])
        
        translate = Node('<LETREC>')
        translate.children.extend([Node('a12'), var, funcbody])
                
        initialize = Node('<APP>')
        initialize.children.extend([Node('a12'), init])
        translate.children.append(initialize)

        return translate
    elif tree.value == '<ASSIGN>':
        new_tree = Node('<LET>')
        new_tree.children.extend(tree.children)
        new_tree.children.append(Node('<UNIT>'))
        return new_tree
    else:
        return tree
