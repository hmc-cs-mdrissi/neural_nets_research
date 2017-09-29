
# coding: utf-8

# In[38]:


# Huffman Tree
#
# Neural Networks Research - September, 2017
# Implemention by Pedro Sandoval
#
import torch.nn as nn


# In[39]:


class Node(object):
    """
    A node for use in a Huffman Tree
    
    Args:
        left: a Node representing left subtree
        right: a Node representing right subtree
        data: a value stored by the Node
        frequency: the frequency at which 'data' occurs in the sequence
    
    Attributes:
        weightVector: 
        biasVector: 
    """
    def __init__(self, left, right, data, frequency, inFeatures, outFeatures = 1):
        self.left = left
        self.right = right
        self.data = data
        self.frequency = frequency
        
        # Every node will contain weight and bias vectors
        self.linear = nn.Linear(inFeatures, outFeatures)

    def isLeaf(self):
        return (self.left == None) and (self.right == None)


# In[40]:


class HuffmanTree(object):
    """
    A Huffman Tree can generate the optimal (shortest length string) path 
    for stored data based on frequencies of occurence
    
    Args:
        tupleList: a list of tuples of the form (data, frequency)
        mean: 
        variance: 
        
    Attributes:
        root: the root Node of the tree
    """
    def __init__(self, tupleList, inFeatures):
        """Initialize a HuffmanTree with a list of tuples 
        containing words and their associated frequencies"""
        
        # Save number of in features
        self.inFeatures = inFeatures
        
        # Convert tuple list to a list of nodes with empty left/right subtrees
        nodeList = [Node(None, None, x[0], x[1], inFeatures) for x in tupleList]
        
        # Find and discard min freq. nodes to make new node
        while len(nodeList) != 1:
            leastNode1, indexToPop = HuffmanTree.__findLeastFrequency(nodeList)
            nodeList.pop(indexToPop)

            leastNode2, indexToPop = HuffmanTree.__findLeastFrequency(nodeList)
            nodeList.pop(indexToPop)

            nodeList.append(self.makeBranch(leastNode1, leastNode2))
            
        # Save root node
        self.root = nodeList[0]

        
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
        #Assume the first tuple is has the minimum amount of occurrences
        minimumFrequency = nodeList[0].frequency
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




