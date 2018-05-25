import torch
import pptree
from enum import IntEnum

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

# TODO: Split make_tree into one function per language. Having a make_tree for all languages is
# going to be problematic as we keep on adding languages.
    
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
        expr = make_tree(children[1], long_base_case=long_base_case,
                         is_lambda_calculus=is_lambda_calculus)
        
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
        unary_operand = make_tree(children[1], long_base_case=long_base_case,
                                  is_lambda_calculus=is_lambda_calculus)
        parentNode.children.extend([Node(unary_op), unary_operand])
        return parentNode

    # Special case for binary operators.
    if tag == '<BINARYOPER>':
        binary_op = "<" + children[1].upper() + ">"
        binary_operand1 = make_tree(children[0], long_base_case=long_base_case,
                                    is_lambda_calculus=is_lambda_calculus)
        binary_operand2 = make_tree(children[2], long_base_case=long_base_case,
                                    is_lambda_calculus=is_lambda_calculus)
        parentNode.children.extend([binary_operand1, Node(binary_op), binary_operand2])
        return parentNode

    if type(children) is list:
        parentNode.children.extend(map(lambda child: make_tree(child, long_base_case=long_base_case,
                                   is_lambda_calculus=is_lambda_calculus), children))
    else:
        parentNode.children.append(make_tree(children, long_base_case=long_base_case,
                                             is_lambda_calculus=is_lambda_calculus))

    return parentNode

EOS = "EOS"

def binarize_tree(tree):
    new_tree = Node(tree.value)
    curr_node = new_tree
    for child in tree.children:
        new_node = binarize_tree(child)
        
        if curr_node is new_tree:
            curr_node.children.append(new_node)
        else:
            curr_node.children[1] = new_node
        
        curr_node = new_node 

    return new_tree

def vectorize(val, num_vars, num_ints, ops, eos_token=False, one_hot=True):
    if val is EOS:
        index = num_vars + num_ints + len(ops.keys)  
    elif type(val) is int:
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

def add_eos(tree, eos, num_children, one_hot=False):
    """
    Add in EOS tokens at the end of all existing branches in a tree
    
    :param tree: the tree which will have eos inserted into it
    :param eos: the EOS value to be inserted (int)
    :param num_children: the maximum number of children a node can have (int)
    :param one_hot: whether to make the values one hot vectors.
    :returns tree: input tree, but with EOS tokens now (also modifies the original tree in-place)
    """
    # Loop through children
    for child in tree.children:
        # Recursively add EOS to children
        add_eos(child, eos, num_children, make_variable)
    
    # Add enough EOS nodes that the final child count is num_children
    while len(tree.children) < num_children:
        if one_hot:
            tree.children.append(Node(make_one_hot(eos + 1, eos)))
        else:
            tree.children.append(Node(eos))

    return tree

def print_tree(tree):
    """
    Print out a tree as a sequence of values
    
    :param tree: the tree to print
    """
    print(int(tree.value))
    for child in tree.children:
        print_tree(child)
    
def t2s_pretty_print_attention(attention_probs, input_tree, target_seq, threshold=0.1):
    """
    Display the parts of the tree focused on at each iteration by the attention mechanism at each 
    step of the generation of the target sequence.
    
    :param attention_probs: attention probabilities; a list of vectors of length equal to the number 
                            of nodes in the input tree
    :param input_tree: input program, in tree form 
    :param target_seq: desired output program, sequence form 
    :param threshold: probability threshold above which we mark the attention as having focused on a 
                      location in the input tree
    """ 
    attention_list = extract_attention(attention_probs, threshold)
    
    # Pretty print each node of the tree
    print("===================== STARTING ATTENTION PRINT =================")
    for i in range(target_seq.size()[0]):
        print("<<<<<<<<")
        # Print the sequence, highlighting the node which was being generated
        pretty_print_seq(target_seq, i)
        # Print the input tree, highlighting the nodes which the attention was focusing on
        pretty_print_attention_tree(attention_list[i], input_tree, None, None, 0)
        print(">>>>>>>>")
    print("===================== ENDING ATTENTION PRINT =================")
    
def pretty_print_seq(target_seq, write_index):
    """
    Print out the sequence as a string, marking the index being generated.
    
    :param target_seq: the desired sequence being generated
    :param write_index: the index of the element in the sequence currently being generated
    """
    
    s = ""
    for i in range(target_seq.size()[0]):
        if i == write_index:
            s += " *" + str(int(target_seq[i])) + "*"
        else:
            s += + " " + str(int(target_seq[i]))
    print(s)
    
def pretty_print_attention(attention_probs, input_tree, threshold=0.1):
    """
    Display the parts of the tree focused on at each iteration by the attention 
    mechanism at each step of the generation of the target sequence. 
    This function was designed for the identity dataset, 
    where the input and target trees are identical.
    
    :param attention_probs: a list of vectors of length equal to the input tree; the 
                            attention mechanism probabilities
    :param input_tree: input program, in tree form 
    :param threshold: probability threshold above which we mark the attention as having focused on a 
                      location in the input tree
    """
    attention_list = extract_attention(attention_probs, threshold)
    
    # Pretty print each node of the tree
    print("===================== STARTING ATTENTION PRINT =================")
    for i in range(target_tree.size()):
        # Mark the nodes being focused on while each node is generated 
        pretty_print_attention_tree(attention_list[i], input_tree, None, i, 0)
    print("===================== ENDING ATTENTION PRINT =================")

def extract_attention(attention_probs, threshold):
    """
    Get a list of the tree locations being focused on at each timestep.
    
    :param attention_probs: attention probabilities; a list of vectors of length equal to the number 
                            of nodes in the input tree
    :param threshold: the probability cutoff to determine whether a node has been focused on by 
                      attention.
    """
    attention_list = []
    # Loop through list (each element corresponding to a diff node in the input tree being 
    # generated)
    for prob in attention_probs:
        important_indices = []
        # Loop through the attention values for one iteration
        for i in range(len(prob)):
            if float(prob[i]) > threshold:
                # Keep track of the indices with high attention probabilities
                important_indices.append(i)
        attention_list.append(important_indices)
    return attention_list

def pretty_print_attention_tree(attention_list, input_tree, parent, write_index, curr_index):
    """
    Display the parts of the tree focused on by the attention 
    while a particular node is being generated.
    This function was designed for the identity dataset, 
    where the input and target trees are identical.
    
    :param attention_list: a list of indices the attention focused on (preorder traversal order)
    :param input_tree: subtree we're currently processing
    :param parent: parent of the node we're currently processing
    :param write_index: node currently being created during this iteration
    :param curr_index: index of the node we're currently processing
    :returns index of one past the last node in our subtree.
    """
    
    # If the current node was being generated or was focused on by the attention, mark it
    root_val = str(int(target_tree.value))
    
    if curr_index == write_index:
        root_val = "*" + root_val + "*"
    if curr_index in attention_list:
        root_val = "(" + root_val + ")"
    
    # Create new node
    root_node = pptree.Node(root_val, parent)
    curr_index += 1
    
    # Recursively add the child subtrees to the tree.
    for child in target_tree.children:
        curr_index = pretty_print_attention_tree(attention_list, child, root_node, write_index, 
                                                 curr_index)
    if parent is None:
        pptree.print_tree(root_node)
        
    return curr_index
        
def pretty_print_tree(tree):
    """
    Print a tree out with a visualized tree structure.
    
    :param tree: the tree to print
    """
    # The printing library uses a special kind of node,
    # so let's just recreate our existing tree as theirs.
    root_node = pptree.Node(str(int(tree.value)))
    for child in tree.children:
        make_pretty_tree(child, root_node)
    
    # Print it out now!
    pptree.print_tree(root_node)
        
def make_pretty_tree(node, parent):
    """
    Helper function for the pretty_print_tree func
    
    :param node: node we are converting into a node of pptree library
    :param parent: parent node (a pptree node)
    """
    new_node = pptree.Node(str(int(node.value)), parent)
    for child in node.children:
        make_pretty_tree(child, new_node)
        
def encode_tree(tree, num_vars, num_ints, ops, eos_token=False, one_hot=True):
    return map_tree(lambda node: vectorize(node, num_vars, num_ints, ops, eos_token=eos_token, 
                                           one_hot=one_hot), tree)

def decode_tokens(seq, num_vars, num_ints, ops):
    reverse_ops = dict(map(lambda p: (p[1], p[0]), ops.items()))

    def index_to_token(index):
        if index < num_ints:
            return index
        elif index < num_ints + num_vars:
            return 'a' + str(index - num_ints)
        elif index == num_ints + num_vars + len(reverse_ops.keys()):
            return EOS
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

class LambdaGrammar(IntEnum):
    EOS = 0
    INT = 1
    VAR_NAME = 2
    VAR = 3
    EXPR = 4
    VARAPPFUNC = 5
    CMP = 6
    TERM = 7
    VARUNITBLANK = 8
    FUNCBLANK = 9
    ALL = 10
    
    
class Lambda(IntEnum):
    VAR = 0
    CONST = 1
    PLUS = 2
    MINUS = 3
    EQUAL = 4
    LE = 5
    GE = 6
    IF = 7
    LET = 8
    UNIT = 9
    LETREC = 10
    APP = 11
    BLANK = 12
    FUNC = 13
    
def parent_to_category_LAMBDA(parent, child_index, num_vars, num_ints):
    """
    Return the category of output which can be produced by a certain parent node 
    at a certain child index in the Lambda language.
    
    :param parent: int, the value of the parent node 
    :param child_index: the index of the child being generated (e.g. left=0, right=1)
    :param num_vars: number of variables a program can use
    :param num_ints: number of ints a program can use
    """
    # If we're at the root of our tree, we can generate mostly anything
    if parent is None:
        return LambdaGrammar.ALL
    
    # If parent is an int or a variable name, we have EOS next
    if int(parent) in range(num_ints + num_vars):
        return LambdaGrammar.EOS
    # If parent is an op, return the class of outputs it can return
    op_index = int(parent) - num_vars - num_ints
    lambda_grammar = {
        Lambda.VAR: [LambdaGrammar.VAR_NAME], 
        Lambda.CONST: [LambdaGrammar.INT], 
        Lambda.PLUS: [LambdaGrammar.EXPR, LambdaGrammar.EXPR],
        Lambda.MINUS: [LambdaGrammar.EXPR, LambdaGrammar.EXPR],
        Lambda.EQUAL: [LambdaGrammar.EXPR, LambdaGrammar.EXPR],
        Lambda.LE: [LambdaGrammar.EXPR, LambdaGrammar.EXPR],
        Lambda.GE: [LambdaGrammar.EXPR, LambdaGrammar.EXPR],
        Lambda.IF: [LambdaGrammar.CMP, LambdaGrammar.TERM, LambdaGrammar.TERM],
        Lambda.LET: [LambdaGrammar.VARUNITBLANK, LambdaGrammar.TERM, LambdaGrammar.TERM],
        Lambda.UNIT: [],
        Lambda.LETREC: [LambdaGrammar.FUNCBLANK, LambdaGrammar.VAR_NAME, LambdaGrammar.TERM, LambdaGrammar.TERM],
#         Lambda.LETREC: [LambdaGrammar.VAR, LambdaGrammar.VAR, LambdaGrammar.TERM, LambdaGrammar.TERM],
        Lambda.APP: [LambdaGrammar.VARAPPFUNC, LambdaGrammar.EXPR],
        Lambda.BLANK: [LambdaGrammar.EOS],
        Lambda.FUNC: [LambdaGrammar.EOS]
    }
    # If we're asking for a child at an index greater than the number of children an op gives,
    # Just return EOS (this happens when an EOS token is a right-hand child)
    if len(lambda_grammar[op_index]) <= child_index:
        return LambdaGrammar.EOS
    return lambda_grammar[op_index][child_index]
    
def category_to_child_LAMBDA(category, num_vars, num_ints):
    """
    Take a category of output, and return a list of new tokens which can be its children in the Lambda language.
    
    :param category: category of output generated next
    :param num_vars: number of variables a program can use
    :param num_ints: number of ints a program can use
    """
    n = num_ints + num_vars
    EOS = num_ints + num_vars + 14 # 14 for the 14 Lambda ops
    lambda_grammar = {
        LambdaGrammar.EOS: [EOS], 
        LambdaGrammar.INT: range(num_ints),
        LambdaGrammar.VAR_NAME: range(num_ints, n),
        LambdaGrammar.VAR: [x + n for x in [Lambda.VAR]],
        LambdaGrammar.EXPR: [x + n for x in [Lambda.VAR, Lambda.CONST, Lambda.PLUS, Lambda.MINUS, Lambda.CONST]],
        LambdaGrammar.VARAPPFUNC: [x + n for x in [Lambda.VAR, Lambda.APP, Lambda.FUNC]],
        LambdaGrammar.CMP: [x + n for x in [Lambda.EQUAL, Lambda.LE, Lambda.GE]],
        LambdaGrammar.TERM: [x + n for x in [Lambda.LET, Lambda.LETREC, Lambda.VAR, Lambda.CONST, Lambda.PLUS, Lambda.MINUS, Lambda.UNIT, Lambda.IF, Lambda.APP, Lambda.BLANK]],
        LambdaGrammar.VARUNITBLANK: [x + n for x in [Lambda.VAR, Lambda.UNIT, Lambda.BLANK]], 
        LambdaGrammar.FUNCBLANK: [x + n for x in [Lambda.FUNC, Lambda.BLANK]],
        LambdaGrammar.ALL: [x + n for x in [Lambda.VAR, Lambda.CONST, Lambda.PLUS, Lambda.MINUS, Lambda.EQUAL, Lambda.LE, Lambda.GE, Lambda.IF, Lambda.LET, Lambda.UNIT, Lambda.LETREC, Lambda.APP]] + [EOS]
    }
    return lambda_grammar[category]


class ForGrammar(IntEnum):
    EOS = 0
    INT = 1
    VAR_NAME = 2
    VAR = 3
    EXPR = 4
    CMP = 5
    SINGLE = 6
    STATEMENT = 7
    ALL = 8
    
    
class For(IntEnum):
    VAR = 0
    CONST = 1
    PLUS = 2
    MINUS = 3
    EQUAL = 4
    LE = 5
    GE = 6
    ASSIGN = 7
    IF = 8
    SEQ = 9
    FOR = 10

def parent_to_category_FOR(parent, child_index, num_vars, num_ints):
    """
    Return the category of output which can be produced by a certain parent node 
    at a certain child index in the For language.
    
    :param parent: int, the value of the parent node 
    :param child_index: the index of the child being generated (e.g. left=0, right=1)
    :param num_vars: number of variables a program can use
    :param num_ints: number of ints a program can use
    """
    # If we're at the root of our tree, we can generate mostly anything
    if parent is None:
        return ForGrammar.ALL
    
    # If parent is an int or a variable name, we have EOS next
    if int(parent) in range(num_ints + num_vars):
        return ForGrammar.EOS
    # If parent is an op, return the class of outputs it can return
    op_index = int(parent) - num_vars - num_ints
    for_grammar = {
        For.VAR: [ForGrammar.VAR_NAME], 
        For.CONST: [ForGrammar.INT], 
        For.PLUS: [ForGrammar.EXPR, ForGrammar.EXPR],
        For.MINUS: [ForGrammar.EXPR, ForGrammar.EXPR],
        For.EQUAL: [ForGrammar.EXPR, ForGrammar.EXPR],
        For.LE: [ForGrammar.EXPR, ForGrammar.EXPR],
        For.GE: [ForGrammar.EXPR, ForGrammar.EXPR],
        For.ASSIGN: [ForGrammar.VAR, ForGrammar.EXPR],
        For.IF: [ForGrammar.CMP, ForGrammar.STATEMENT, ForGrammar.STATEMENT],
        For.SEQ: [ForGrammar.STATEMENT, ForGrammar.SINGLE],
        For.FOR: [ForGrammar.VAR_NAME, ForGrammar.EXPR, ForGrammar.CMP, ForGrammar.EXPR, ForGrammar.STATEMENT]
    }
    # If we're asking for a child at an index greater than the number of children an op gives,
    # Just return EOS (this happens when an EOS token is a right-hand child)
    if len(for_grammar[op_index]) <= child_index:
        return ForGrammar.EOS
    return for_grammar[op_index][child_index]
    
def category_to_child_FOR(category, num_vars, num_ints):
    """
    Take a category of output, and return a list of new tokens which can be its children in the For language.
    
    :param category: category of output generated next
    :param num_vars: number of variables a program can use
    :param num_ints: number of ints a program can use
    """
    n = num_ints + num_vars
    EOS = num_ints + num_vars + 11 #11 for the 11 For ops
    for_grammar = {
        ForGrammar.EOS: [EOS], 
        ForGrammar.INT: range(num_ints),
        ForGrammar.VAR_NAME: range(num_ints, n),
        ForGrammar.VAR: [x + n for x in [For.VAR]],
        ForGrammar.EXPR: [x + n for x in [For.VAR, For.CONST, For.PLUS, For.MINUS]],
        ForGrammar.CMP: [x + n for x in [For.EQUAL, For.LE, For.GE]],
        ForGrammar.SINGLE: [x + n for x in [For.ASSIGN, For.IF, For.FOR]],
        ForGrammar.STATEMENT: [x + n for x in [For.ASSIGN, For.IF, For.FOR, For.SEQ]],
        ForGrammar.ALL: [x + n for x in [For.VAR, For.CONST, For.PLUS, For.MINUS, For.EQUAL, For.LE, For.GE, For.ASSIGN, For.IF, For.SEQ, For.FOR]] + [EOS]
    }
    return for_grammar[category]

    
def translate_from_for(tree):
    if tree.value == '<SEQ>':
        t1 = translate_from_for(tree.children[0])
        t2 = translate_from_for(tree.children[1])
        if t1.value == '<LET>' and t1.children[-1].value == '<UNIT>':
            t1.children[-1] = t2
            return t1
        else:
            new_tree = Node('<LET>')
            new_tree.children.extend([Node('<BLANK>'), t1, t2])
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
        tb.children.append(Node('<BLANK>'))
        tb.children.append(body)
        increment = Node('<APP>')
        increment.children.extend([Node('<FUNC>'), inc])
        tb.children.append(increment)

        funcbody = Node('<IF>')
        funcbody.children.extend([cmp, tb, Node('<UNIT>')])
        
        translate = Node('<LETREC>')
        translate.children.extend([Node('<FUNC>'), var, funcbody])
                
        initialize = Node('<APP>')
        initialize.children.extend([Node('<FUNC>'), init])
        translate.children.append(initialize)

        return translate
    elif tree.value == '<ASSIGN>':
        new_tree = Node('<LET>')
        new_tree.children.extend(tree.children)
        new_tree.children.append(Node('<UNIT>'))
        return new_tree
    else:
        return tree
 

