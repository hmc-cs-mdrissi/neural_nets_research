import torch
import torch.utils.data as data

class DummyDataset(data.Dataset):
    def __init__(self, M, num_examples):
        """
        Program: [[1,0],[0,1]]
        
        :param M: The allowable range of integers (from 0 to M-1)
        :param num_examples: The number of training examples to be generated
        """
        self.input_list = []
        self.output_list = []
        
        for i in range(num_examples):
            
            initial_memory = torch.zeros(M, M)
            output_memory = torch.zeros(M, M)
            
            # Set the initial memory
            for i in range(M):
                output_memory[i][i] = 1
                initial_memory[i][i] = 1
            
            # Output mask is length of the list itself
            output_mask = torch.ones(M, M)
            
            self.input_list.append(initial_memory)
            self.output_list.append((output_memory, output_mask))
       
    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, i):
        """
        Get the i^th element of the dataset.
        
        :param i: The index of the element to be returned.
        :return A tuple containing i^th element of the dataset.
        """
        return self.input_list[i], self.output_list[i]