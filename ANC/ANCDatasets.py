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

class TrivialAccessTaskDataset(data.Dataset):
    def __init__(self, M, num_examples):
        """
        Generate a dataset for the access task by randomly generating an array.
        The task is to access the 3rd element of the array

        :param M: The allowable range of integers (from 0 to M-1)
        :param num_examples: The number of training examples to be generated
        """
        self.input_list = []

        for i in range(num_examples):

            initial_memory = torch.zeros(M, M)
            output_memory = torch.zeros(M, M)

            # Set the initial memory
            for i in range(1,M):
                list_val = random.randint(0, M-1)
                initial_memory[i][list_val] = 1

            # Get 3rd element of array
            initial_memory[0, 3] = 1
            output_memory[0, 4] = 1

            # Output mask is length of the list itself
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

class TrivialAddTaskDataset(data.Dataset):
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
            initial_memory = torch.FloatTensor(M, M).zero_()
            initial_memory[0][first_addend] = 1
            initial_memory[1][second_addend] = 1
            for j in range(2, M):
                initial_memory[j][0] = 1


            output_memory = torch.FloatTensor(M, M).zero_()
            output_memory[0][(first_addend + second_addend) % M] = 1

            # Output mask has ones in the rows of the memory matrix where the answer will be stored.
            output_mask = torch.FloatTensor(M, M).zero_()
            output_mask[2] = torch.ones(M)

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

class IncrementTaskDataset(data.Dataset):
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


