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

def general_base_cases(json):
    # First base case - variable name
    if isinstance(json, string_types):
        return Node(make_var_name(json))

    # Second base case - variable value
    if type(json) is int:
        return Node(json)
    
    return None

def make_tree_for(json, long_base_case=True):
    check_general_base_cases = general_base_cases(json)
    
    if check_general_base_cases is not None:
        return check_general_base_cases
    
    tag = "<" + json["tag"].upper() + ">"
    parentNode = Node(tag)
    
    children = json["contents"]
    
    if not long_base_case:
        if tag == '<CONST>' or tag == '<VAR>':
            return Node(children)
    
    # Special case for assignment.
    if tag == '<ASSIGN>':
        var_name = children[0]
        expr = make_tree_for(children[1], long_base_case=long_base_case)
        
        if long_base_case:
            var = Node('<VAR>')
            var.children.append(Node(var_name))
        else:
            var = Node(var_name)
        
        parentNode.children.extend([var, expr])
        return parentNode
    
    if type(children) is list:
        parentNode.children.extend(map(lambda child: 
                                       make_tree_for(child, long_base_case=long_base_case), 
                                       children))
    else:
        parentNode.children.append(make_tree_for(children, long_base_case=long_base_case))

    return parentNode
    
def make_tree_lambda(json, long_base_case=True):
    return make_tree_lambda_coffee(json, long_base_case=long_base_case)
    
def make_tree_lambda_coffee(json, long_base_case=True):
    check_general_base_cases = general_base_cases(json)
    
    if check_general_base_cases is not None:
        return check_general_base_cases
    
    tag = "<" + json["tag"].upper() + ">"
    parentNode = Node(tag)
    
    children = json["contents"]
    
    if not long_base_case:
        if tag == '<CONST>' or tag == '<VAR>':
            return Node(children)
        
    if type(children) is list:
        parentNode.children.extend(map(lambda child: 
                                       make_tree_lambda(child, long_base_case=long_base_case), 
                                       children))
    else:
        parentNode.children.append(make_tree_lambda(children, long_base_case=long_base_case))

    return parentNode 
    
def make_tree_lambda_calculus(json, long_base_case=True):
    check_general_base_cases = general_base_cases(json)
    
    if check_general_base_cases is not None:
        return check_general_base_cases
    
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
        if tag == '<NUMBER>' or tag == '<VARIABLE>':
            return Node(children)
        
    # Special case for unary operators.
    if tag == '<UNARYOPER>':
        unary_op = "<" + children[0].upper() + ">"
        unary_operand = make_tree_lambda_calculus(children[1], long_base_case=long_base_case)
        parentNode.children.extend([Node(unary_op), unary_operand])
        return parentNode

    # Special case for binary operators.
    if tag == '<BINARYOPER>':
        binary_op = "<" + children[1].upper() + ">"
        binary_operand1 = make_tree_lambda_calculus(children[0], long_base_case=long_base_cases)
        binary_operand2 = make_tree_lambda_calculus(children[2], long_base_case=long_base_case)
        parentNode.children.extend([binary_operand1, Node(binary_op), binary_operand2])
        return parentNode
    
    if type(children) is list:
        parentNode.children.extend(map(lambda child: 
                                       make_tree_lambda_calculus(child, 
                                                                 long_base_case=long_base_case), 
                                       children))
    else:
        parentNode.children.append(make_tree_lambda_calculus(children, 
                                                             long_base_case=long_base_case))

    return parentNode    

# TODO: All of the make_tree's with pass.
def make_tree_javascript(json, long_base_case=True):
    
    check_general_base_cases = general_base_cases(json)
    
    # Ignore these keys completely
    ignore_words = ["start", "end", "init", "generator", "computed", "sourceType", "kind", "type", "operator"]
    
    # Base cases (ints and var names)
    if check_general_base_cases is not None:
        return check_general_base_cases
    
    # Set node's value
    tag = "<" + json["type"].upper() + ">"
    parentNode = Node(tag)
    
    # Handle special cases
        
    # Variable names
    if tag == "<IDENTIFIER>":
        if long_base_case:
            parentNode.children.append(Node(json["name"]))
            return parentNode
        else: 
            return Node(json["name"])

    # Literal
    if tag == "<LITERAL>":
        if long_base_case:
            parentNode.children.append(Node(json["value"]))
            return parentNode
        else:
            return Node(json["value"])

    # Binary expression
    if tag == "<BINARYEXPRESSION>":
        parentNode = Node(json["operator"]) #Alternatively the operator could be a child of this node

    for key in json.keys():
        if not key in ignore_words:

            children = json[key]
            if isinstance(children, list):
                parentNode.children.extend(map(lambda child: make_tree_javascript(child, long_base_case=long_base_case), children))
            # Chop out null/false branches #TODO: check that false is actually ignore-able
            elif children:
                parentNode.children.append(make_tree_javascript(children, long_base_case=long_base_case))
    return parentNode 

def make_tree_coffeescript(json, long_base_case=True):
    return make_tree_lambda_coffee(json, long_base_case=long_base_case)

def make_tree_java(json):
    pass # IMPLEMENTED IN DATASET PREPROCESSING 'CUZ IT'S MORE CONVENIENT, WILL BE TRANSFERRED WHEN DONE

def make_tree_csharp(json, long_base_case=True):
    print(json)
    
    if general_base_cases(json) is not None:
        return general_base_cases(json)
    
    # There should really only be one
    for key in json.keys():
        
        tag = "<" + key.upper() + ">"
        verbose_tokens = ["<CHARACTERLITERALEXPRESSION>", "<NUMERICLITERALEXPRESSION>", "<STRINGLITERALEXPRESSION>", "<IDENTIFIERNAME>"]
        if not long_base_case and tag in verbose_tokens:
            return make_tree_csharp(json[key])
        else:
            parentNode = Node(tag)
            children = json[key]
            if type(children) is list:
                parentNode.children.extend(map(lambda child: 
                                       make_tree_csharp(child, long_base_case=long_base_case), 
                                       children))
            else:
                parentNode.children.append(make_tree_csharp(children, long_base_case=long_base_case))
    return parentNode 

def canonicalize_csharp(tree):
    num_names = {}
    var_names = {}
    char_names = {}
    string_names = {}
    
    def make_generic(node, dict, symbol):
        if node.value in dict:
            node.value = dict[node.value]
        else:
            new_symbol = symbol + str(len(dict) + 1)
            dict[node.value] = new_symbol
            node.value = new_symbol

    
    def canonicalize_csharp_helper(tree):
        if tree.value == "<NUMERICLITERALEXPRESSION>":
            make_generic(tree.children[0], num_names, "n")
        elif tree.value == "<CHARACTERLITERALEXPRESSION>":
            make_generic(tree.children[0], var_names, "c")
        elif tree.value == "<STRINGLITERALEXPRESSION>":
            make_generic(tree.children[0], string_names, "s")
        elif tree.value == "<IDENTIFIERNAME>":
            make_generic(tree.children[0], var_names, "v") #TODO: deal with argumentlist
        else:
            for child in tree.children:
                canonicalize_csharp_helper(child)
                
    canonicalize_csharp_helper(tree)
    return tree

# TODO: Canonicalizing trees for java/csharp.
def canonicalize_java(tree):
    pass # IMPLEMENTED IN DATASET PREPROCESSING 'CUZ IT'S MORE CONVENIENT, WILL BE TRANSFERRED WHEN DONE

def binarize_tree(tree):
    """
    Binarize a tree using the left-child right-sibling representation.
    To deal with the issue of a node potentially having a right child, but not a left child,
    None is used. None is fine for the input tree. For the output tree, None should be replaced
    by EOS and add_eos will take care of that. Some extra Nones may end up present and can
    be cleaned out with clean_binarized_tree. Cleaning only matters for the input tree
    as any extra Nones would end up replaced by EOS anyway.
    
    :param tree: the tree which to be binarized    
    """
    new_tree = Node(tree.value)
    new_tree.children = [Node(None), Node(None)]
    curr_node = new_tree
    
    for child in tree.children:
        new_node = binarize_tree(child)
                
        if curr_node is new_tree:
            curr_node.children[0] = new_node
        else:
            curr_node.children[1] = new_node
        curr_node = new_node 
    
    return new_tree

def clean_binarized_tree(tree):
    if len(tree.children) == 2:
        if tree.children[0].value is None and tree.children[1].value is None:
            tree.children = []
        elif tree.children[1].value is None:
            tree.children = [tree.children[0]]
    elif len(tree.children) == 1:
        if tree.children[0].value is None:
            tree.children = []
            
    for child in tree.children:
        clean_binarized_tree(child)
    
    return tree

EOS = "EOS"

def vectorize(val, num_vars, num_ints, ops, eos_token=False, one_hot=False): 
    """
        Based on the value, num_variables, num_ints, and the possible ops, the index corresponding
        to the value is found. value should not correspond to the eos_token. Instead vectorization
        should occur prior to adding eos_tokens. Nodes with value None are simply returned as None.
    """
    if val == EOS:
        if not eos_token:
            raise ValueError("EOS tokens should not be present while eos_token is false")
        
        index = num_ints + num_vars + len(ops.keys())
    elif type(val) is int:
        index = val % num_ints
    elif val not in ops:
        index = int(val[1:]) % num_vars + num_ints
    else:
        index = num_ints + num_vars + ops[val]

    if one_hot:
        eos_bonus = 1 if eos_token else 0
        return make_one_hot(num_vars + num_ints + len(ops.keys()) + eos_bonus, index)

    return torch.LongTensor([index])

def make_one_hot(len, index):
    vector = torch.zeros(len)
    vector[index] = 1
    return vector

def un_one_hot(vector):
    return int(vector.nonzero())

def map_tree(func, tree):
    new_tree = Node(func(tree.value) if tree.value is not None else tree.value)
    new_tree.children.extend(map(partial(map_tree, func), tree.children))
    return new_tree

def add_eos(program, num_children=None):
    """
    Add in EOS tokens at the end of all existing branches in a tree or to end of sequence as
    needed.
    
    :param program: the program which will have eos inserted into it
    :param num_children: the maximum number of children a node can have (int). Only needed
                         if you are doing this on a tree.
    :returns program: input program, but with EOS tokens now (also modifies the original in-place)
    """
    if isinstance(program, Node):            
        return add_eos_tree(program, num_children)
    else:
        program = list(program)
        program.append(EOS)
        return program

def add_eos_tree(tree, num_children):
    # Loop through children
    for i, child in enumerate(tree.children):
        if child.value is None:
            tree.children[i] = Node(EOS)
        else:
            # Recursively add EOS to children
            add_eos(tree.children[i], num_children)

    # Add enough EOS nodes that the final child count is num_children
    while len(tree.children) < num_children:
        tree.children.append(Node(EOS))

    return tree
        
def print_tree(tree):
    """
    Print out a tree as a sequence of values
    
    :param tree: the tree to print
    """
    if tree.value is not None:
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
    
def pretty_print_attention_t2t(attention_probs, input_tree, target_tree, threshold=0.1):
    """
    Display the parts of the tree focused on at each iteration by the attention 
    mechanism at each step of the generation of the target sequence. 
    This function was designed for the identity dataset, 
    where the input and target trees are identical.
    
    :param attention_probs: a list of vectors of length equal to the input tree; the attention 
                            mechanism probabilities
    :param input_tree: input program, in tree form 
    :param target_tree: target program, in tree form 
    :param threshold: probability threshold above which we mark the attention as having focused on a 
                      location in the input tree
    """
    attention_list = extract_attention(attention_probs, threshold)
    
    # Pretty print each node of the tree
    print("===================== STARTING ATTENTION PRINT =================")
    for i in range(target_tree.size()):
        # Mark the nodes being focused on while each node is generated 
        print(">>>")
        pretty_print_attention_tree(attention_list[i], input_tree, None, -1, 0) #Input
        pretty_print_attention_tree([], target_tree, None, i, 0) # output
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
    pptree.print_tree(map_tree(lambda val: str(get_val(val)), tree), nameattr="value")
    
def get_val(value):
    """
    Extract the integer value of the input (or keep it as a string it it's not an integer/tensor)
    
    :param value: an integer or tensor (with one integer element)
    """
    if type(value) is torch.Tensor:
        return int(value)
    else:
        return value
    
def encode_program(program, num_vars, num_ints, ops, eos_token=False, one_hot=False):
    if isinstance(program, Node):
        return map_tree(lambda node: vectorize(node, num_vars, num_ints, ops, eos_token=eos_token, 
                                               one_hot=one_hot), program)
    else:
        program = map(lambda node: vectorize(node, num_vars, num_ints, ops, eos_token=eos_token, 
                                             one_hot=one_hot), program)
        if one_hot:
            return torch.stack(list(program))
        else:
            return torch.LongTensor(list(program))

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


class CoffeeGrammar(IntEnum):
    INT = 0
    VAR_NAME = 1
    TERMINAL = 2
    EXPR = 3
    SIMPLE = 4
    IF_TYPE = 5
    WHILE_TYPE = 6
    BLOCK = 7
    SHORT_STATEMENT = 8
    STATEMENT = 9
    
    
def category_to_child_coffee(num_vars, num_ints, category):
    """
    Take a category of output, and return a list of new tokens which can be its children in the For 
    language.
    
    :param num_vars: number of variables a program can use
    :param num_ints: number of ints a program can use
    :param category: category of output generated next
    """
    n = num_ints + num_vars
    coffee_grammar = {
        CoffeeGrammar.INT: range(num_ints),
        CoffeeGrammar.VAR_NAME: range(num_ints, n),
        CoffeeGrammar.EXPR: [x + n for x in [Coffee.PLUS, Coffee.TIMES, Coffee.EQUAL, Coffee.VAR, Coffee.CONST]],
        CoffeeGrammar.TERMINAL: [x + n for x in [Coffee.VAR, Coffee.CONST]],
        CoffeeGrammar.SIMPLE: [x + n for x in [Coffee.ASSIGN, Coffee.EXPR]],
        CoffeeGrammar.IF_TYPE: [x + n for x in [Coffee.IFSIMPLE, Coffee.IFCOMPLEX]],
        CoffeeGrammar.WHILE_TYPE: [x + n for x in [Coffee.WHILESIMPLE, Coffee.WHILECOMPLEX]],
        CoffeeGrammar.BLOCK: [x + n for x in [Coffee.SIMPLECS, Coffee.COMPLEXCS]],
        CoffeeGrammar.SHORT_STATEMENT: [x + n for x in [Coffee.SIMPLESTATEMENT, Coffee.SIMPLEIF, Coffee.SIMPLEWHILE]],
        CoffeeGrammar.STATEMENT: [x + n for x in [Coffee.SHORTSTATEMENTCS, Coffee.IFTHENELSE, Coffee.IFELSE, Coffee.WHILE, Coffee.IF]],
    }
    
    return coffee_grammar[category]

    
class Coffee(IntEnum):
    VAR = 0
    CONST = 1
    PLUS = 2
    TIMES = 3
    EQUAL = 4
    ASSIGN = 5
    IF = 6
    IFSIMPLE = 7
    SIMPLEIF = 8
    IFELSE = 9
    IFTHENELSE = 10
    IFCOMPLEX = 11
    SIMPLECS = 12
    COMPLEXCS = 13
    EXPR = 14
    SHORTSTATEMENTCS = 15
    WHILE = 16
    WHILESIMPLE = 17
    SIMPLEWHILE = 18
    WHILECOMPLEX = 19
    SIMPLESTATEMENT = 20
    ROOT = 21
    
    
def parent_to_category_coffee(num_vars, num_ints, parent):
    """
    Return the categories of output which can be produced by a certain parent node.
    
    :param num_vars: number of variables a program can use
    :param num_ints: number of ints a program can use
    :param parent: int, the value of the parent node 
    """
    
    # If parent is an int or a variable name, we are done.
    if int(parent) in range(num_ints + num_vars):
        return []
    
    # If parent is an op, return the class of outputs it can return
    op_index = int(parent) - num_vars - num_ints
    coffee_grammar = { 
        Coffee.VAR: [CoffeeGrammar.VAR_NAME],
        Coffee.CONST: [CoffeeGrammar.INT],
        Coffee.PLUS: [CoffeeGrammar.EXPR, CoffeeGrammar.EXPR],
        Coffee.TIMES: [CoffeeGrammar.EXPR, CoffeeGrammar.EXPR],
        Coffee.EQUAL: [CoffeeGrammar.EXPR, CoffeeGrammar.EXPR],
        Coffee.ASSIGN: [CoffeeGrammar.VAR_NAME, CoffeeGrammar.EXPR],
        Coffee.IF: [CoffeeGrammar.EXPR, CoffeeGrammar.BLOCK],
        Coffee.IFSIMPLE: [CoffeeGrammar.SIMPLE, CoffeeGrammar.EXPR],
        Coffee.SIMPLEIF: [CoffeeGrammar.IF_TYPE],
        Coffee.IFELSE: [CoffeeGrammar.EXPR, CoffeeGrammar.BLOCK, CoffeeGrammar.BLOCK],
        Coffee.IFTHENELSE: [CoffeeGrammar.EXPR, CoffeeGrammar.SHORT_STATEMENT, CoffeeGrammar.SHORT_STATEMENT],
        Coffee.IFCOMPLEX: [CoffeeGrammar.IF_TYPE, CoffeeGrammar.EXPR],
        Coffee.SIMPLECS: [CoffeeGrammar.STATEMENT],
        Coffee.COMPLEXCS: [CoffeeGrammar.BLOCK, CoffeeGrammar.STATEMENT],
        Coffee.EXPR: [CoffeeGrammar.EXPR],
        Coffee.SHORTSTATEMENTCS: [CoffeeGrammar.SHORT_STATEMENT],
        Coffee.WHILE: [CoffeeGrammar.EXPR, CoffeeGrammar.BLOCK],
        Coffee.WHILESIMPLE: [CoffeeGrammar.SIMPLE, CoffeeGrammar.EXPR],
        Coffee.SIMPLEWHILE: [CoffeeGrammar.WHILE_TYPE],
        Coffee.WHILECOMPLEX: [CoffeeGrammar.WHILE_TYPE, CoffeeGrammar.EXPR],
        Coffee.SIMPLESTATEMENT: [CoffeeGrammar.SIMPLE],
    }
    
    op_index = int(parent) - num_vars - num_ints
    
    # Special case for the root
    if op_index == Coffee.ROOT:
        return [CoffeeGrammar.BLOCK]
    
    # If parent is an int or a variable name, we are done.
    if int(parent) < num_ints + num_vars:
        return []
    return coffee_grammar[op_index]
    
    

class LambdaGrammar(IntEnum):
    INT = 0
    VAR_NAME = 1
    VAR = 2
    EXPR = 3
    VARAPP = 4
    CMP = 5
    TERM = 6
    VARUNIT = 7
    
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
    ROOT = 12

def parent_to_category_LAMBDA(num_vars, num_ints, parent):
    """
    Return the categories of output which can be produced by a certain parent index.
    
    :param num_vars: number of variables a program can use
    :param num_ints: number of ints a program can use
    :param parent: int, the value of the parent node 
    """    
    lambda_grammar = {
        Lambda.ROOT: [LambdaGrammar.TERM],
        Lambda.VAR: [LambdaGrammar.VAR_NAME], 
        Lambda.CONST: [LambdaGrammar.INT], 
        Lambda.PLUS: [LambdaGrammar.EXPR, LambdaGrammar.EXPR],
        Lambda.MINUS: [LambdaGrammar.EXPR, LambdaGrammar.EXPR],
        Lambda.EQUAL: [LambdaGrammar.EXPR, LambdaGrammar.EXPR],
        Lambda.LE: [LambdaGrammar.EXPR, LambdaGrammar.EXPR],
        Lambda.GE: [LambdaGrammar.EXPR, LambdaGrammar.EXPR],
        Lambda.IF: [LambdaGrammar.CMP, LambdaGrammar.TERM, LambdaGrammar.TERM],
        Lambda.LET: [LambdaGrammar.VARUNIT, LambdaGrammar.TERM, LambdaGrammar.TERM],
        Lambda.UNIT: [],
        Lambda.LETREC: [LambdaGrammar.VAR_NAME, LambdaGrammar.VAR_NAME, LambdaGrammar.TERM, 
                        LambdaGrammar.TERM],
        Lambda.APP: [LambdaGrammar.VARAPP, LambdaGrammar.EXPR]
    }
    
    op_index = int(parent) - num_vars - num_ints
    
    # Special case for the root
    if op_index == Lambda.ROOT:
        return lambda_grammar[Lambda.ROOT]
    
    # If parent is an int or a variable name, we are done.
    if int(parent) < num_ints + num_vars:
        return []
    
    return lambda_grammar[op_index]
    
def category_to_child_LAMBDA(num_vars, num_ints, category):
    """
    Take a category of output, and return a list of new tokens which can be its children in the 
    Lambda language.
    
    :param num_vars: number of variables a program can use
    :param num_ints: number of ints a program can use
    :param category: category of output generated next
    """
    n = num_ints + num_vars
    lambda_grammar = {
        LambdaGrammar.INT: range(num_ints),
        LambdaGrammar.VAR_NAME: range(num_ints, n),
        LambdaGrammar.VAR: [x + n for x in [Lambda.VAR]],
        LambdaGrammar.EXPR: [x + n for x in [Lambda.VAR, Lambda.CONST, Lambda.PLUS, Lambda.MINUS, Lambda.CONST]],
        LambdaGrammar.VARAPP: [x + n for x in [Lambda.VAR, Lambda.APP]] + list(range(num_ints, n)),
        LambdaGrammar.CMP: [x + n for x in [Lambda.EQUAL, Lambda.LE, Lambda.GE]],
        LambdaGrammar.TERM: [x + n for x in [Lambda.LET, Lambda.LETREC, Lambda.PLUS, Lambda.MINUS, Lambda.VAR,
                                             Lambda.CONST, Lambda.UNIT, Lambda.IF, Lambda.APP]],
        LambdaGrammar.VARUNIT: [x + n for x in [Lambda.VAR]] + list(range(num_ints, n)),
    }
    
    return lambda_grammar[category]

class ForGrammar(IntEnum):
    INT = 1
    VAR_NAME = 2
    VAR = 3
    EXPR = 4
    CMP = 5
    SINGLE = 6
    STATEMENT = 7
    
class For(IntEnum):
    ROOT = 0
    VAR = 1
    CONST = 2
    PLUS = 3
    MINUS = 4
    EQUAL = 5
    LE = 6
    GE = 7
    ASSIGN = 8
    IF = 9
    SEQ = 10
    FOR = 11

def parent_to_category_FOR(num_vars, num_ints, parent):
    """
    Return the categories of output which can be produced by a certain parent node.
    
    :param num_vars: number of variables a program can use
    :param num_ints: number of ints a program can use
    :param parent: int, the value of the parent node 
    """
    
    # If parent is an int or a variable name, we are done.
    if int(parent) in range(num_ints + num_vars):
        return []
    
    # If parent is an op, return the class of outputs it can return
    op_index = int(parent) - num_vars - num_ints
    for_grammar = {
        For.ROOT: [ForGrammar.STATEMENT],
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
        For.FOR: [ForGrammar.VAR_NAME, ForGrammar.EXPR, ForGrammar.CMP, ForGrammar.EXPR, 
                  ForGrammar.STATEMENT]
    }
    
    return for_grammar[op_index]
    
def category_to_child_FOR(num_vars, num_ints, category):
    """
    Take a category of output, and return a list of new tokens which can be its children in the For 
    language.
    
    :param num_vars: number of variables a program can use
    :param num_ints: number of ints a program can use
    :param category: category of output generated next
    """
    n = num_ints + num_vars
    for_grammar = {
        ForGrammar.INT: range(num_ints),
        ForGrammar.VAR_NAME: range(num_ints, n),
        ForGrammar.VAR: [x + n for x in [For.VAR]],
        ForGrammar.EXPR: [x + n for x in [For.VAR, For.CONST, For.PLUS, For.MINUS]],
        ForGrammar.CMP: [x + n for x in [For.EQUAL, For.LE, For.GE]],
        ForGrammar.SINGLE: [x + n for x in [For.ASSIGN, For.IF, For.FOR]],
        ForGrammar.STATEMENT: [x + n for x in [For.ASSIGN, For.IF, For.FOR, For.SEQ]],
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
            new_tree.children.extend([Node('a8'), t1, t2])
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
        tb.children.append(Node('a8'))
        tb.children.append(body)
        increment = Node('<APP>')
        increment.children.extend([Node('a9'), inc])
        tb.children.append(increment)

        funcbody = Node('<IF>')
        funcbody.children.extend([cmp, tb, Node('<UNIT>')])
        
        translate = Node('<LETREC>')
        translate.children.extend([Node('a9'), var, funcbody])
                
        initialize = Node('<APP>')
        initialize.children.extend([Node('a9'), init])
        translate.children.append(initialize)

        return translate
    elif tree.value == '<ASSIGN>':
        new_tree = Node('<LET>')
        new_tree.children.extend(tree.children)
        new_tree.children.append(Node('<UNIT>'))
        return new_tree
    else:
        return tree
