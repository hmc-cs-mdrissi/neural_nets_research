"""
Huffman Tree
Neural Networks Research - September, 2017
Implemention by Pedro Sandoval
"""
import torch.nn as nn
from queue import PriorityQueue

class Node(object):
    """
    A node for use in a Huffman Tree

    Args:
        left: a Node representing left subtree
        right: a Node representing right subtree
        data: a value stored by the Node
        frequency: the frequency at which 'data' occurs in the sequence
        hidden_size: The dimension of the word representation.
    """
    def __init__(self, left, right, parent, data, frequency, hidden_size):
        self.left = left
        self.right = right
        self.parent = parent
        self.data = data
        self.frequency = frequency

        # Every node will contain weight and bias vectors
        self.linear = nn.Linear(hidden_size, 1)

    def is_leaf(self):
        """Returns true if the node is a leaf."""
        return self.left is None and self.right is  None
    
    def __lt__(self, other):
        """Implementation of < operator for queue sorting purposes"""
        return self.frequency < other.frequency

class HuffmanTree(object):
    """
    A Huffman Tree can generate the optimal (shortest length string) path
    for stored data based on frequencies of occurence

    Args:
        tuple_list: A list of tuples of the form (data, frequency)
        hidden_size: The dimension of the word representation.

    Attributes:
        root: The root node of the tree
        data_to_path: A dictionary containing data to path mappings
    """
    def __init__(self, tuple_list, hidden_size):
        """Initialize a HuffmanTree with a list of tuples
        containing words and their associated frequencies"""

        # Save dimension of word representations.
        self.hidden_size = hidden_size

        # Convert tuple list to a list of nodes with empty left/right subtrees
        node_list = [Node(None, None, None, x[0], x[1], hidden_size) for x in tuple_list]

        # Enqueue all nodes with priority
        priority_queue = PriorityQueue()
        for node in node_list:
            priority_queue.put((node.frequency, node))

        # Find and discard min freq. nodes to make new node
        while priority_queue.qsize() != 1:
            _, least_node1 = priority_queue.get()
            _, least_node2 = priority_queue.get()

            new_node = self.make_branch(least_node1, least_node2)
            priority_queue.put((new_node.frequency, new_node))

        # Save root node
        _, self.root = priority_queue.get()
        
        # Load dictionary with all paths to data
        self.data_to_path = dict()
        for leaf in self.get_leaf_nodes():
            self.data_to_path[leaf.data] = HuffmanTree.__get_path_for_node(leaf)
        
    def make_branch(self, left_node, right_node):
        """Create a new node with children"""
        new_frequency = left_node.frequency + right_node.frequency
        new_node = Node(left_node, right_node, None, None, new_frequency, self.hidden_size)
        left_node.parent = new_node; right_node.parent = new_node;
        return new_node
    
    def get_leaf_nodes(self):
        """Get all the leaf nodes containing data"""
        return HuffmanTree.__get_leaf_nodes_helper(self.root)
    
    @staticmethod
    def __get_leaf_nodes_helper(root):
        if root.is_leaf():
            return [root]

        return HuffmanTree.__get_leaf_nodes_helper(root.left) + \
               HuffmanTree.__get_leaf_nodes_helper(root.right)
    
    def get_path(self, data):
        """Get the prefix code (path) for data in the tree"""
        return self.data_to_path[data]

    @staticmethod
    def __get_path_for_node(node):
        """Travel up tree from leaf node to compute path"""
        path = ""
        while node.parent != None:
            current_parent = node.parent
            if current_parent.left == node:
                path = "0" + path
            else:
                path = "1" + path

            node = current_parent

        return path
    
    def get_modules_on_path(self, data):
        """Get the linear modules along a path for data in tree"""
        path = self.get_path(data)

        list_modules = []
        current_node = self.root

        for char in path:
            list_modules += [current_node.linear]
            if char == '0':
                current_node = current_node.left
            else:
                current_node = current_node.right

        return list_modules

    def get_modules(self):
        """Get all modules of each node"""
        return HuffmanTree.__get_modules_helper(self.root)

    @staticmethod
    def __get_modules_helper(root):
        if root.is_leaf():
            return [root.linear]

        return [root.linear] + HuffmanTree.__get_modules_helper(root.left) + \
                               HuffmanTree.__get_modules_helper(root.right)
