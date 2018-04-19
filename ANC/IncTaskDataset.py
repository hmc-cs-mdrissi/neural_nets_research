class IncTaskDataset(data.Dataset):
    def __init__(self, M, list_len, num_examples):
        """
        Generate a dataset for the list task by randomly choosing two numbers in the allowed range
        and creating the initial/final matrices for adding them.
        
        :param M: The allowable range of integers (from 0 to M-1)
        :param list_len: The list length
        :param num_examples: The number of training examples to be generated
        """
        
        if list_len > M:
            raise ValueError("Cannot have a list longer than M")
        
        self.input_list = []
        self.output_list = []
        
        for i in range(num_examples):
#             list_val = random.randint(1, M-1)
            list_val = 2#i % M
            initial_memory = torch.zeros(M, M)
            output_memory = torch.zeros(M, M)
            # Output mask is length of the list itself
            output_mask = torch.zeros(M, M)
            
            for i in range(list_len):
                initial_memory[i][list_val] = 1
                output_memory[i][(list_val + 1 ) % M] = 1
                output_mask[i] = torch.ones(M)
                
            for j in range(list_len, M):
                initial_memory[j][0] = 1
            
#             self.input_list.append((initial_memory, output_memory, output_mask))
            self.input_list.append(initial_memory)
#             print("IM", initial_memory)
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