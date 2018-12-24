import torch
from six import string_types
import json
import sys
sys.path.append('../')
from tree_to_sequence.translating_trees import Node, general_base_cases

def make_tree_math(json, big_tree=True):
    
    if not big_tree:
        return make_tree_math_short(json)
    
    # Base case for variable names, symbols, or numbers
    base_case = general_base_cases(json)
    if (base_case):
        return base_case
    
    # Base case for empty lists (we just ignore these)
    if json == []:
        return []
    
    parentNode = Node(json["tag"])
    
    # Base case for Nil
    if not "contents" in json:
        return parentNode
    
    children = json["contents"]
    if children != []:
        if type(children) is list:
            parentNode.children.extend(
                map(lambda child: make_tree_math(child), children))
        else:
            single_child = make_tree_math(children)
            if not single_child == []:
                parentNode.children.append(single_child)
    return parentNode


def make_tree_math_short(json):    
    # Base case for variable names, symbols, or numbers
    base_case = general_base_cases(json)
    if (base_case):
        return base_case
    
    value = json["tag"]
    parent_node = Node(value)
    
    # Base case for Nil
    if not "contents" in json:
        return parent_node
    children = json["contents"]
        
    if value == "Digit":
        parent_node = make_tree_math_short(children[1])
        parent_node.children = [make_tree_math_short(children[0])]
        return parent_node

    # Don't include "IntegerM", "VarName", or "Symbol" tokens
    if value in ["IntegerM", "VarName", "Symbol"]:
        return make_tree_math_short(children)

    # Don't use UnOp, UnOpExp, or DoubOp tokens.  Instead make the first child the parent
    if value in ["UnOp", "UnOpExp", "DoubOp"]:
        parent_node = make_tree_math_short(children[0])
        parent_node.children.extend(
                map(lambda child: make_tree_math_short(child), children[1:]))
        return parent_node


    # Don't use PUnOp or BinOp tokens.  Instead mak the second child the parent
    if value in ["PUnOp", "BinOp"]:
        parent_node = make_tree_math_short(children[1])
        parent_node.children.extend(map(lambda child: 
              make_tree_math_short(child), [children[0]] + children[2:]))
        return parent_node

    # For containers, ignore braces entirely.  For the others, make the container the parent
    # and make its content the children.
    if value == "Container":
        firstChild = children[0]
        # Ignore braces since they're invisible
        if firstChild == "LeftBrace":
            return make_tree_math_short(children[1])

        # Otherwise, come up with a name for the container
        name_map = {
            "AbsBar": "Abs",
            "LeftParen": "Parens",
            "Magnitude": "Magnitude",
        }

        container_name = name_map[firstChild]
        parent_node = Node(container_name)
        parent_node.children.append(make_tree_math_short(children[1]))
    else:
        if type(children) is list:
            parent_node.children.extend(
                map(lambda child: make_tree_math_short(child), children))
        else:
            parent_node.children.append(make_tree_math_short(children))
    return parent_node



math_tokens_short = {
    "Nil": 0,
    "Sum": 1,
    "Integral": 2,
    "AbsBar": 3,
    "Parens": 4,
    "Magnitude": 5,
    "Plus": 6,
    "Minus": 7,
    "Div": 8,
    "Mult": 9,
    "BinaryPm": 10,
    "Equal": 11,
    "Marrow": 12,
    "SubscriptOp": 13,
    "SuperscriptOp": 14,
    "ImplicitMult": 15,
    "Le": 16,
    "Leq": 17,
    "Ge": 18,
    "Geq": 19,
    "Neq": 20,
    "Sin": 21,
    "Cos": 22,
    "Tan": 23,
    "Sqrt": 24,
    "NegSign": 25,
    "UnaryPm": 26,
    "FracOp": 27,
    "LogOp": 28,
    "LimOp": 29,
    "Alpha": 30,
    "Beta": 31,
    "Gamma": 32,
    "Phi": 33,
    "Pi": 34,
    "Theta": 35,
    "Infty": 36,
    "Ldots": 37,
    "Factorial": 28,
}


# math_tokens_short = {
#     "FracOp": 0,    
#     "ImplicitMult": 1
# }

# math_tokens = {
#     "IntegerM": 0,
#     "Digit": 1,
#     "VarName": 2,
#     "Nil": 3,
#     "Symbol": 4,
#     "Container": 5,
#     "UnOp": 6,
#     "UnOpExp": 7,
#     "PUnOp": 8,
#     "DoubOp": 9,
#     "BinOp": 10,
#     "Sum": 11,
#     "Integral": 12,
#     "AbsBar": 13,
#     "LeftParen": 14,
#     "RightParen": 15,
#     "LeftBrace": 16,
#     "RightBrace": 17,
#     "Magnitude": 18,
#     "Plus": 19,
#     "Minus": 20,
#     "Div": 21,
#     "Mult": 22,
#     "BinaryPm": 23,
#     "Equal": 24,
#     "Marrow": 25,
#     "SubscriptOp": 26,
#     "SuperscriptOp": 27,
#     "ImplicitMult": 28,
#     "Le": 29,
#     "Leq": 30,
#     "Ge": 31,
#     "Geq": 32,
#     "Neq": 33,
#     "Sin": 34,
#     "Cos": 35,
#     "Tan": 36,
#     "Sqrt": 37,
#     "NegSign": 38,
#     "UnaryPm": 39,
#     "FracOp": 40,
#     "LogOp": 41,
#     "LimOp": 42,
#     "Alpha": 43,
#     "Beta": 44,
#     "Gamma": 45,
#     "Phi": 46,
#     "Pi": 47,
#     "Theta": 48,
#     "Infty": 49,
#     "Ldots": 50,
#     "Factorial": 51,
    
# }
