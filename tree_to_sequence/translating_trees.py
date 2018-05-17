import torch
from torch.autograd import Variable
import pptree

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
    
    def size(self):
        return 1 + sum([child.size() for child in self.children])

def make_var_name(var_name):
    if var_name == 'h':
        return '<HEAD>'
    elif var_name == 't':
        return '<TAIL>'
    else:
        return var_name

def make_tree(json, long_base_case=True, is_lambda_calculus=False):
    # First base case - variable name
    if isinstance(json, string_types):
        return Node(make_var_name(json))

    # Second base case - variable value
    if type(json) is int:
        return Node(json)

    # Third base case for head of abstraction.
    if type(json) is list:
        var_type = "<" + json[1]["tag"].upper() + ">"
        var_name = make_var_name(json[0])

        parentNode = Node("<ARGUMENT>")
        parentNode.children.extend([Node(var_type), Node(var_name)])
        return parentNode

    # Fourth base case for booleans.
    if type(json) is bool:
        bool_str = "<" + str(json).upper() + ">"
        return Node(bool_str)

    tag = "<" + json["tag"].upper() + ">"
    parentNode = Node(tag)

    # Fifth base for nil.
    if tag == '<NIL>':
        return parentNode

    children = json["contents"]
    
    if not long_base_case:
        if tag == '<CONST>' or tag == '<NUMBER>' or tag == '<VARIABLE>' or tag == '<VAR>':
            return Node(children)
    
    # Special case for assignment.
    if tag == '<ASSIGN>':
        var_name = children[0]
        expr = make_tree(children[1], long_base_case=long_base_case, is_lambda_calculus=is_lambda_calculus)
        
        if long_base_case:
            var = Node('<VAR>')
            var.children.append(Node(var_name))
        else:
            var = Node(var_name)
        
        parentNode.children.extend([var, expr])
        return parentNode

    # Special case for unary operators.
    if tag == '<UNARYOPER>':
        unary_op = "<" + children[0].upper() + ">"
        unary_operand = make_tree(children[1], long_base_case=long_base_case, is_lambda_calculus=is_lambda_calculus)
        parentNode.children.extend([Node(unary_op), unary_operand])
        return parentNode

    # Special case for binary operators.
    if tag == '<BINARYOPER>':
        binary_op = "<" + children[1].upper() + ">"
        binary_operand1 = make_tree(children[0], long_base_case=long_base_case, is_lambda_calculus=is_lambda_calculus)
        binary_operand2 = make_tree(children[2], long_base_case=long_base_case, is_lambda_calculus=is_lambda_calculus)
        parentNode.children.extend([binary_operand1, Node(binary_op), binary_operand2])
        return parentNode

    if type(children) is list:
        parentNode.children.extend(map(lambda child: make_tree(child, long_base_case=long_base_case, is_lambda_calculus=is_lambda_calculus), children))
    else:
        parentNode.children.append(make_tree(children, long_base_case=long_base_case, is_lambda_calculus=is_lambda_calculus))

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
        index = val % num_ints
    elif val not in ops:
        index = int(val[1:]) + num_ints
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
    print(get_val(tree.value))
    for child in tree.children:
        print_tree(child)
        
def get_val(value):
    if type(value) is torch.autograd.variable.Variable:
        return value.data[0]
    else:
        return value
    
def t2s_pretty_print_attention(attention_probs, input_tree, target_seq, threshold=0.1):
    attention_list = extract_attention(attention_probs, threshold)
    
    # Pretty print each node of the tree
    print("===================== STARTING ATTENTION PRINT =================")
    for i in range(target_seq.size()[0]):
        print("<<<<<<<<")
        pretty_print_seq(target_seq, i)
        pretty_print_attention_tree(attention_list[i], input_tree, None, None, 0)
        print(">>>>>>>>")
    print("===================== ENDING ATTENTION PRINT =================")
    
def pretty_print_seq(target_seq, write_index):
    s = ""
    for i in range(target_seq.size()[0]):
        if i == write_index:
            s = s + " *" + str(int(target_seq[i].data)) + "*"
        else:
            s = s + " " + str(int(target_seq[i].data))
    print(s)
    
def pretty_print_attention(attention_probs, target_tree, threshold=0.1):
    attention_list = extract_attention(attention_probs, threshold)
    
    # Pretty print each node of the tree
    print("===================== STARTING ATTENTION PRINT =================")
    for i in range(target_tree.size()):
        pretty_print_attention_tree(attention_list[i], target_tree, None, i, 0)
    print("===================== ENDING ATTENTION PRINT =================")

def extract_attention(attention_probs, threshold):
    attention_list = []
    for prob in attention_probs:
        important_indices = []
        for i in range(len(prob)):
            if float(prob[i]) > threshold:
                important_indices.append(i)
        attention_list.append(important_indices)
    return attention_list


def pretty_print_attention_tree(attention_list, target_tree, parent, write_index, curr_index):
    root_val = str(get_val(target_tree.value))
    if curr_index == write_index:
        root_val = "*" + root_val + "*"
    if curr_index in attention_list:
        root_val = "(" + root_val + ")"
    if parent is None:
        root_node = pptree.Node(root_val, parent)
    else:
        root_node = pptree.Node(root_val, parent)
    curr_index += 1
    for child in target_tree.children[::]:
        curr_index = pretty_print_attention_tree(attention_list, child, root_node, write_index, curr_index)
    if parent is None:
        pptree.print_tree(root_node)
    return curr_index
        
def pretty_print_tree(tree):
    root_node = pptree.Node(str(get_val(tree.value)))
    for child in tree.children:
        make_pretty_tree(child, root_node)
    pptree.print_tree(root_node)
        
def make_pretty_tree(node, parent):
    new_node = pptree.Node(str(get_val(node.value)), parent)
    for child in node.children:
        make_pretty_tree(child, new_node)
        

def encode_tree(tree, num_vars, num_ints, ops, eos_token=False, one_hot=True):
    return map_tree(lambda node: vectorize(node, num_vars, num_ints, ops, eos_token=eos_token, one_hot=one_hot), tree)

def decode_tokens(seq, num_vars, num_ints, ops):
    reverse_ops = dict(map(lambda p: (p[1], p[0]), ops.items()))

    def index_to_token(index):
        if index < num_ints:
            return index
        elif index < num_ints + num_vars:
            return 'a' + str(index - num_ints)
        else:
            return reverse_ops[index - num_ints - num_vars]

    return list(map(lambda val: index_to_token(int(val)), seq))

def tree_to_list(tree):
  """
        Concatenate a tree into a list using a pre-order traversal.

        :param tree: a tree.
        :return a list of values of the tree
  """
  return [tree.value] + list(itertools.chain.from_iterable(map(tree_to_list, tree.children)))

# Non-binarized
def get_for_grammar():
    expr = [0, 1, 2, 3] # (var, const, plus, minus)
    cmp = [4, 5, 6] # (==, <, >)
    single = [7, 8, 10] # (Assign, If, For)
    for_grammar = {
        0: [-1], # Var aaaaaaaa
        1: [-2], # Const -> value
        2: [expr, expr], # Plus -> (expr), (expr)
        3: [expr, expr], # Minus -> (expr), (expr)
        4: [expr, expr], # Equals -> (expr), (expr)
        5: [expr, expr], # <= -> (expr), (expr)
        6: [expr, expr], # >= -> (expr), (expr)
        7: [[0], expr], # Assign -> (var), (expr)
        8: [cmp, single + [9], []], # If -> (cmp) (statement) (statement)
        9: [],
        10: [],
    }
    
    for_ops = {
    "Var": 0,
    "Const": 1,
    "Plus": 2,
    "Minus": 3,
    "EqualFor": 4,
    "LeFor": 5,
    "GeFor": 6,
    "Assign": 7,
    "If": 8,
    "Seq": 9,
    "For": 10
}
    
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
 
