import torch
from torch.autograd import Variable

from functools import partial
import itertools

class Node:
    """
    Node class
    """
    def __init__(self, value):
        self.value = value
        self.children = []

def make_tree(json):
  # First base case - variable name
  if type(json) is str:
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

def vectorize(val, num_vars, num_ints, ops):
    vector = torch.zeros(num_vars + num_ints + len(ops.keys()))

    if type(val) is int:
        vector[val] = 1
    elif val not in ops:
        vector[int(val[1:]) + num_ints] = 1
    else:
        index = ops[val]
        vector[num_ints + num_vars + index] = 1

    return vector

def map_tree(func, tree):
  new_tree = Node(func(tree.value))
  new_tree.children.extend(map(partial(map_tree, func), tree.children))
  return new_tree

def print_tree(tree):
    print(tree.value)
    for child in tree.children:
        print_tree(child)

def encode_tree(tree, num_vars, num_ints, ops):
  return map_tree(lambda node: Variable(vectorize(node, num_vars, num_ints, ops)), tree)

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
