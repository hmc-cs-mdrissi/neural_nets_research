import torch
import torch.utils.data as data

class AddTaskDataset(data.Dataset):
    def __init__(self, M, num_examples):
        """
        Generate a dataset for the addition task by randomly choosing two numbers in the allowed range
        and creating the initial/final matrices for adding them.
        
        :param M: The allowable range of integers (from 0 to M-1)
        :param num_examples: The number of training examples to be generated
        """
        
        self.input_list = []
        
        for i in range(num_examples):
            first_addend = random.randint(0, M-1)
            second_addend = random.randint(0, M-1)
            initial_memory = torch.zeros(M, M)
            initial_memory[0][first_addend] = 1
            initial_memory[1][second_addend] = 1
            for j in range(2, M):
                initial_memory[j][0] = 1

            
            output_memory = torch.zeros(M, M)
            output_memory[0][(first_addend + second_addend) % M] = 1

            # Output mask has ones in the row of the memory matrix where the answer will be stored.
            output_mask = torch.zeros(M, M)
            output_mask[0] = torch.ones(M)
            
            self.input_list.append((initial_memory, output_memory, output_mask))
       
    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, i):
        """
        Get the i^th element of the dataset.
        
        :param i: The index of the element to be returned.
        :return A tuple containing i^th element of the dataset.
        """
        return self.input_list[i]