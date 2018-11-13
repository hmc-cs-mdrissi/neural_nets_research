import torch
from six import string_types
import json
import sys
sys.path.append('../')
from tree_to_sequence.translating_trees import Node, general_base_cases

def make_tree_math(json, big_tree=False):
    
    # Base case for variable names, symbols, or numbers
    base_case = general_base_cases(json)
    if (base_case):
        if not isinstance(json, string_types):
            print(json)
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
                map(lambda child: make_tree_math(child, big_tree=big_tree), children))
        else:
            single_child = make_tree_math(children, big_tree=big_tree)
            if not single_child == []:
                parentNode.children.append(single_child)
    return parentNode

# def make_tree_math(json, big_tree=False):
    
#     # Base case for variable names, symbols, or numbers
#     base_case = general_base_cases(json)
#     if (base_case):
#         return base_case
    
#     # Base case for empty lists (we just ignore these)
#     if json == []:
#         return []
    
#     value = json["tag"]
#     parent_node = Node(value)
    
#     # Base case for Nil
#     if not "contents" in json:
#         return parent_node
#     children = json["contents"]
    
#     # For condensed tree...
#     if big_tree == False:
#         # Don't include "IntegerM", "VarName", or "Symbol" tokens
#         if value in ["IntegerM", "VarName", "Symbol"]:
#             return make_tree_math(children, big_tree=big_tree)
        
#         # Don't use UnOp, UnOpExp, or DoubOp tokens.  Instead make the first child the parent
#         if value in ["UnOp", "UnOpExp", "DoubOp"]:
#             parent_node = make_tree_math(children[0], big_tree=big_tree)
#             parent_node.children.extend(
#                     map(lambda child: make_tree_math(child, big_tree=big_tree), children[1:]))
            
            
#         # Don't use PUnOp or BinOp tokens.  Instead mak the second child the parent
#         if value in ["PUnOp", "BinOp"]:
#             parent_node = make_tree_math(children[1], big_tree=big_tree)
#             parent_node.children.extend(map(lambda child: 
#                   make_tree_math(child, big_tree=big_tree), [children[0]] + children[2:]))
            
#         # For containers, ignore braces entirely.  For the others, make the container the parent
#         # and make its content the children.
#         if value == "Container":
#             firstChild = children[0]
#             # Ignore braces since they're invisible
#             if firstChild == "LeftBrace":
#                 return make_tree_math(children[1], big_tree=big_tree)
            
#             # Otherwise, come up with a name for the container
#             name_map = {
#                 "AbsBar": "Abs",
#                 "LeftParen": "Parens",
#                 "Magnitude": "Magnitude",
#             }
            
#             container_name = name_map[firstChild]
#             parent_node = Node(container_name)
#             parent_node.children.append(make_tree_math(children[1], big_tree=big_tree))
#         else:
#             if type(children) is list:
#                 parent_node.children.extend(
#                     map(lambda child: make_tree_math(child, big_tree=big_tree), children))
#             else:
#                 parent_node.children.append(make_tree_math(children, big_tree=big_tree))
#     return parent_node



math_tokens = {
    "IntegerM": 0,
    "DoubleM": 1,
    "VarName": 2,
    "Nil": 3,
    "Symbol": 4,
    "Container": 5,
    "UnOp": 6,
    "UnOpExp": 7,
    "PUnOp": 8,
    "DoubOp": 9,
    "BinOp": 10,
    "Sum": 11,
    "Integral": 12,
    "AbsBar": 13,
    "LeftParen": 14,
    "RightParen": 15,
    "LeftBrace": 16,
    "RightBrace": 17,
    "Magnitude": 18,
    "Plus": 19,
    "Minus": 20,
    "Div": 21,
    "Mult": 22,
    "BinaryPm": 23,
    "Equal": 24,
    "Marrow": 25,
    "SubscriptOp": 26,
    "SuperscriptOp": 27,
    "ImplicitMult": 28,
    "Le": 29,
    "Leq": 30,
    "Ge": 31,
    "Geq": 32,
    "Neq": 33,
    "Sin": 34,
    "Cos": 35,
    "Tan": 36,
    "Sqrt": 37,
    "NegSign": 38,
    "UnaryPm": 39,
    "FracOp": 40,
    "LogOp": 41,
    "LimOp": 42,
    "Alpha": 43,
    "Beta": 44,
    "Gamma": 45,
    "Phi": 46,
    "Pi": 47,
    "Theta": 48,
    "Infty": 49,
    "Ldots": 50,
    "Factorial": 51,
    "Digit": 52,
}
