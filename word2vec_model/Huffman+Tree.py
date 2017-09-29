# Huffman Tree
#
# Neural Networks Research - September, 2017
# Implemention by Pedro Sandoval
#
import torch.nn as nn


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
        return self.left is None and self.right is  None

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

        # Find and discard min freq. nodes to make new node
        while len(node_list) != 1:
            least_node1, index_to_pop = HuffmanTree.__findLeastFrequency(node_list)
            node_list.pop(index_to_pop)

            least_node2, index_to_pop = HuffmanTree.__findLeastFrequency(node_list)
            node_list.pop(index_to_pop)

            node_list.append(self.makeBranch(least_node1, least_node2))

        # Save root node
        self.root = node_list[0]


    def getPath(self, data):
        """Get the prefix code (path) for data in the tree"""
        return HuffmanTree.__getPathHelper(self.root, data)

    def getModulesOnPath(self, data):
        """Get the linear modules along a path for data in tree"""
        path = self.getPath(data)

        listModules = []
        currentNode = self.root
        for index in range(len(path)):
            listModules += [currentNode.linear]
            if path[index] == '0':
                currentNode = currentNode.left
            else:
                currentNode = currentNode.right

        return listModules

    def getModules(self):
        """Get all modules of each node"""
        return HuffmanTree.__getModulesHelper(self.root)

    def __getModulesHelper(root):
        if root.isLeaf():
            return [root.linear]
        else:
            return [root.linear] + HuffmanTree.__getModulesHelper(root.left) + HuffmanTree.__getModulesHelper(root.right)

    def isDataInTree(self, data):
        """Return true if word is in tree"""
        return HuffmanTree.__isDataInSubtree(self.root, data)

    def __findLeastFrequency(nodeList):
        """Get the node with the minimum frequency from a list of nodes"""
        # Initially, assume the first tuple has the minimum amount of occurrences
        minimumFrequency = node_list[0].frequency
        minimumIndex = 0

        for index in range(len(nodeList)):
            currentFrequency = nodeList[index].frequency
            if currentFrequency < minimumFrequency:
                minimumFrequency = currentFrequency
                minimumIndex = index

        return nodeList[minimumIndex], minimumIndex

    def makeBranch(self, leftNode, rightNode):
        """Create a new node with children"""
        newFrequency = leftNode.frequency + rightNode.frequency
        return Node(leftNode, rightNode, None, newFrequency, self.inFeatures)

    def __isDataInSubtree(root, data):
        """Returns true if word is in subtree"""
        if root.isLeaf():
            return root.data == data
        else:
            return HuffmanTree.__isDataInSubtree(root.left, data) or HuffmanTree.__isDataInSubtree(root.right, data)

    def __getPathHelper(root, data):
        """See getPath(self, data)"""
        if root.data == data:
            return ""
        elif HuffmanTree.__isDataInSubtree(root.left, data):
            return "0" + HuffmanTree.__getPathHelper(root.left, data)
        else:
            return "1" + HuffmanTree.__getPathHelper(root.right, data)


# In[41]:


# Test Case
tupleList = [("a", 7), ("b", 5),("c", 10),("d", 2),("e", 789)]

# The tree should look like:
#
#                  o
#                 / \
#               o   [e]
#              / \
#           [c]   o
#                / \
#             [a]   o
#                  / \
#               [d]  [b]
#
# where data occurring most frequently has the shortest path encoding

huffman = HuffmanTree(tupleList, 5)
print(huffman.isDataInTree('b'))       # 'b' is in the tree
print(huffman.getPath("d"))            # the path to 'd' is (left, right, right, left) or 0110

print(huffman.getModulesOnPath('c'))   # returns modules along a path
print(huffman.getModules())            # returns all modules in the tree


# In[ ]:
