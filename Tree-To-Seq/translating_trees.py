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

def make_tree(json):
  # First base case - variable name
  if isinstance(json, string_types):
      parentNode = Node("<VAR>")
      childNode = Node(json)
      parentNode.children.append(childNode)
      return parentNode

  # Second base case - variable value
  if type(json) is int:
      return Node(json)

  tag = "<" + json["tag"].upper() + ">"
  children = json["contents"]
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

def vectorize(val, num_vars, num_ints, ops, one_hot=True):
    if type(val) is int:
        index = val
    elif val not in ops:
        index = int(val[1:]) - 1 + num_ints
    else:
        index = num_ints + num_vars + ops[val]

    if one_hot:
      vector = torch.zeros(num_vars + num_ints + len(ops.keys()))
      vector[index] = 1
      return Variable(vector)

    return index

def map_tree(func, tree):
  new_tree = Node(func(tree.value))
  new_tree.children.extend(map(partial(map_tree, func), tree.children))
  return new_tree

def print_tree(tree):
    print(tree.value)
    for child in tree.children:
        print_tree(child)

def encode_tree(tree, num_vars, num_ints, ops, one_hot=True):
  return map_tree(lambda node: vectorize(node, num_vars, num_ints, ops, one_hot=one_hot), tree)

def tree_to_list(tree):
  """
        Concatenate a tree of vectors into a matrix using a pre-order traversal.

        :param node: a tree of vectors, each of the same size.
        :return a list of vectors of the tree
  """
  return [tree.value] + list(itertools.chain.from_iterable(map(tree_to_list, tree.children)))

def translate_from_for(tree):
    if tree.value == '<SEQ>':
        t1 = self.translate_from_for(tree.children[0])
        t2 = self.translate_from_for(tree.children[1])
        if t1.value == '<LET>' and t1.children[-1].value == '<UNIT>':
            t1.children[-1] = t2
            return t1
        else:
            new_tree = Node('<LET>')
            new_tree.children.extend(['blank', t1, t2])
            return new_tree
    elif tree.value == '<IF>':
        cmp = ls.children[0]
        t1 = self.translate_from_for(tree.children[1])
        t2 = self.translate_from_for(tree.children[2])
        new_tree = Node('<IF>')
        new_tree.children.extend([cmp, t1, t2])
        return new_tree
    elif tree.value == '<FOR>':
        var = tree.children[0]
        init = self.translate_from_for(tree.children[1])
        cmp = self.translate_from_for(tree.children[2])
        inc = self.translate_from_for(tree.children[3])
        body = self.translate_from_for(tree.children[4])

        tb = Node('<LET>')
        tb.children.append(Node('blank'))
        tb.children.append(body)
        increment = Node('<APP>')
        increment.children.extend([Node('func'), inc])
        tb.children.append(increment)

        funcbody = Node('<IF>')
        funcbody.children.extend([cmp, tb, Node('<UNIT>')])

        translate = Node('<LETREC>')
        translate.children.extend([Node('func'), var, funcbody])
        initialize = Node('<APP>')
        initialize.children.extend([Node('func'), init])
        translate.append(initialize)

        return translate
    elif tree.value == '<ASSIGN>':
        new_tree = Node('<LET>')
        new_tree.children.extend(tree.children)
        new_tree.children.append(Node('<UNIT>'))
        return new_tree
    else:
        return tree
