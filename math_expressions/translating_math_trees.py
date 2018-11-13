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
        return base_case
    
    # Base case for empty lists (we just ignore these)
    if json == []:
        return []
    
    parentNode = Node(json["tag"])
    
    # Base case for Nil
    if not "contents" in json:
        return parentNode
    
    children = json["contents"]
    if type(children) is list:
        parentNode.children.extend(
            map(lambda child: make_tree_math(child, big_tree=big_tree), children))
    else:
        parentNode.children.append(make_tree_math(children, big_tree=big_tree))
    return parentNode



math_ops = {
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
}
