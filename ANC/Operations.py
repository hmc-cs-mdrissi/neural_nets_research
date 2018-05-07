import torch.nn as nn
import torch
from torch.autograd import Variable

class Operation(nn.Module):
    """
    Parent class for our binary operations
    """
    def __init__(self, M):
        """
        Initialize the memory length (needed so we can mod our answer in case it exceeds the range 0-M-1)
        Also calculate the output matrix for the operation
        
        :param M: Memory length
        """
        super(Operation, self).__init__()
        self.M = M
        
        # Create a MxMxM matrix where the (i,j,k) cell is 1 iff operation(i,j) = k.
        self.outputs = torch.IntTensor(M, M, M).zero_()
        for i in range(M):
            for j in range(M):
                val = self.compute(i, j)
                self.outputs[i][j][val] = 1
                    
    def compute(self, x, y):
        """ 
        Perform the binary operation.  The arguments may or may not be used.
        
        :param x: First argument
        :param y: Second argument
        """
        raise NotImplementedError
    
    def forward(self):
        """
        :return: The output matrix
        """
        return self.outputs


class Add(Operation):

    def __init__(self, M):
        super(Add, self).__init__(M)
    
    def compute(self, x, y):
        return (x + y) % self.M


class Stop(Operation):
    
    def __init__(self, M):
        super(Stop, self).__init__(M)

    def compute(self, _1, _2):
        return 0


class Jump(Operation):
    
    def __init__(self, M):
        super(Jump, self).__init__(M)

    def compute(self, _1, _2):
        return 0 # Actual jump happens in the Machine class

class Decrement(Operation):
    
    def __init__(self, M):
        super(Decrement, self).__init__(M)

    def compute(self, x, _):
        return (x - 1) % self.M


class Increment(Operation):
    
    def __init__(self, M):
        super(Increment, self).__init__(M)

    def compute(self, x, _):
        return (x + 1) % self.M


class Max(Operation):
    
    def __init__(self, M):
        super(Max, self).__init__(M)

    def compute(self, x, y):
        return max(x,y)


class Min(Operation):
    
    def __init__(self, M):
        super(Min, self).__init__(M)

    def compute(self, x, y):
        return min(x,y)


class Read(Operation):
    
    def __init__(self, M):
        super(Read, self).__init__(M)
        # Leave output matrix blank since we're gonna do the reading elsewhere
        self.outputs = torch.zeros(M, M, M)

    def compute(self, x, _):
        return 0 # Actual reading happens in the Machine class


class Subtract(Operation):
    
    def __init__(self, M):
        super(Subtract, self).__init__(M)

    def compute(self, x, y):
        return (x - y) % self.M


class Write(Operation):
    
    def __init__(self, M):
        super(Write, self).__init__(M)

    def compute(self, x, y):
        return 0 # Actual write happens in the Machine class


class Zero(Operation):
    
    def __init__(self, M):
        super(Zero, self).__init__(M)

    def compute(self, _1, _2):
        return 0





















