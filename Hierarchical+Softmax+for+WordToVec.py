
# coding: utf-8
import torch


class HierarchicalSoftmaxTree(nn.Module):
    def __init__(self, huffmanTree):
        self.huffmanTree = huffmanTree
        self.modules = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()
        self.root = huffmanTree.root
        
        for weight in huffmanTree.getModules():
            self.modules.append(modules)
    
# inputWordVec is the input word
# path is the path from of 1's and 0's from the huffman tree

    def forward_on_path(self, inputWordVec, desiredOutput):
        modulesOnPath = self.huffmanTree.getModulesOnPath(desiredOutput)
        #Probably will error out for number of parameters
        probability = Variable(torch.ones(1), requires_grad = True)
        for module in modulesOnPath:
            branchProbability = self.sigmoid(module(inputWordVec))
            probability *= branchProbability
        return probability
            
    def forward(self, inputWordVec, id_list):
        probabilityList = []
        for batch in id_list:
            probabilityList.append([self.forward_on_path(inputWordVec, desiredOutput) for desiredOutput in batch])
        return torch.FloatTensor(probabilityList)
        
    def backProp_on_path(self, desiredOutput, learningRate):
        modulesOnPath = self.huffmanTree.getModulesOnPath(desiredOutput)
        for module in modulesOnPath:
            for p in module.parameters():
                p -= learningRate * p.grad
                # zero gradients after we make the calculation
                p.zero_grad()
                
        
    def backProp(inputWordVec, id_list, learningRate):
        for batch in id_list:
            for desiredOutput in batch:
                self.backProp_on_path(desiredOutput, learningRate)
                
    
    def NLLL(probabilityList):
        sumVar = Variable(torch.zeros(1), requires_grad(True))
        count = 0.0
        for batch in probabilityList:
            for probability in batch:
                sumVar += -1 * torch.log(probability)
                count += 1
                
        return sumVar / count
                
