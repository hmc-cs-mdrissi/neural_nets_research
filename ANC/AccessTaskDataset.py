import torch
import torch.utils.data as data

class AccessTaskDataset(data.Dataset):
    def __init__(self, M, num_examples):
        """
        Generate a dataset for the access task by randomly generating an array.
        The task is to access the 3rd element of the array
        
        :param M: The allowable range of integers (from 0 to M-1)
        :param num_examples: The number of training examples to be generated
        """
        self.input_list = []
        self.output_list = []
        
        for i in range(num_examples):
            
            initial_memory = torch.zeros(M, M)
            output_memory = torch.zeros(M, M)
            
            # Set the initial memory
            for i in range(1,M):
                list_val = random.randint(0, M-1)
                initial_memory[i][list_val] = 1
                
                if i == 4:
                    output_memory[0, list_val] = 1
            
            # Get 3rd element of array
            initial_memory[0, 3] = 1
            
            # Output mask is length of the list itself
            output_mask = torch.zeros(M, M)
            output_mask[0] = torch.ones(M)
            
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