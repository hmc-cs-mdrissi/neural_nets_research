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
    def __init__(self, left, right, data, frequency, hidden_size):
        self.left = left
        self.right = right
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
    """
    def __init__(self, tuple_list, hidden_size):
        """Initialize a HuffmanTree with a list of tuples
        containing words and their associated frequencies"""

        # Save dimension of word representations.
        self.hidden_size = hidden_size

        # Convert tuple list to a list of nodes with empty left/right subtrees
        node_list = [Node(None, None, x[0], x[1], hidden_size) for x in tuple_list]

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

    def make_branch(self, left_node, right_node):
        """Create a new node with children"""
        new_frequency = left_node.frequency + right_node.frequency
        return Node(left_node, right_node, None, new_frequency, self.hidden_size)
    
    def get_path(self, data):
        """Get the prefix code (path) for data in the tree"""
        return HuffmanTree.__get_path_helper(self.root, data)

    @staticmethod
    def __get_path_helper(root, data):
        """See getPath(self, data)"""
        if root.data == data:
            return ""

        if HuffmanTree.__is_data_in_subtree(root.left, data):
            return "0" + HuffmanTree.__get_path_helper(root.left, data)

        return "1" + HuffmanTree.__get_path_helper(root.right, data)

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

    def is_data_in_tree(self, data):
        """Return true if word is in tree"""
        return HuffmanTree.__is_data_in_subtree(self.root, data)

    @staticmethod
    def __is_data_in_subtree(root, data):
        """Returns true if word is in subtree"""
        if root.is_leaf():
            return root.data == data

        return HuffmanTree.__is_data_in_subtree(root.left, data) or \
               HuffmanTree.__is_data_in_subtree(root.right, data)

    @staticmethod
    def __find_least_frequency(node_list):
        """Get the node with the minimum frequency from a list of nodes"""

        # Initially, assume the first tuple has the minimum amount of occurrences
        minimum_frequency = node_list[0].frequency
        minimum_index = 0

        for index, node in enumerate(node_list):
            current_frequency = node.frequency

            if current_frequency < minimum_frequency:
                minimum_frequency = current_frequency
                minimum_index = index

        return node_list[minimum_index], minimum_index


