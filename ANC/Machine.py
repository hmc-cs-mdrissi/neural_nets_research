import torch.nn as nn
from torch.autograd import Variable
from .Operations import *

class Machine(nn.Module):
    """
    The Machine executes assembly instructions passed to it by the Controller.
    It updates the given memory, registers, and instruction pointer.
    The Machine doesn't have any learnable parameters.
    """
    def __init__(self, M, R):
        """
        Initializes dimensions, operations, and counters
        
        :param M: Memory length.  Integer values also take on values 0-M-1.  M is also the program length.
        :param R: Number of registers
        """
        super(Machine, self).__init__()
        
        # Store parameters as class variables
        self.R = R # Number of registers
        self.M = M # Memory length (also largest number)
        
        # Start off with 0 probability of stopping
        self.stop_probability = 0 
        
        # List of ops (must be in same order as the original ANC paper so compilation works right)
        self.ops = [ 
            Stop(M),
            Zero(M),
            Increment(M),
            Add(M),
            Subtract(M),
            Decrement(M),
            Min(M),
            Max(M),
            Read(M),
            Write(M),
            Jump(M)
        ]
        
        # Number of instructions
        self.N = len(self.ops)
        
        # Create a 4D matrix composed of the output matrices of each of the ops
        self.register_buffer('outputs', torch.zeros(self.N, self.M, self.M, self.M))
        
        for i in range(self.N):
            op = self.ops[i]
            self.outputs[i] = op()
                
        # Keep track of ops which will be handled specially
        self.jump_index = 10
        self.stop_index = 0
        self.write_index = 9
        self.read_index = 8 
        
    def forward(self, e, a, b, o, memory, registers, IR):
        
        """
        Run the Machine for one timestep (corresponding to the execution of one line of Assembly).
        The first four parameter names correspond to the vector names used in the original ANC paper
        
        :param e: Probability distribution over the instruction being executed (N)
        :param a: Probability distribution over the first argument register (length R)
        :param b: Probability distribution over the second argument register (length R)
        :param o: Probability distribution over the first argument register (length R)
        :param memory: Memory matrix (size MxM)
        :param registers: Register matrix (size RxM)
        :param IR: Instruction Register (length M)
        
        :return: The memory, registers, and instruction register after the timestep
        """

        # Calculate distributions over the two argument values by multiplying each 
        # register by the probability that register is being used.
        arg1 = torch.matmul(a, registers)
        arg2 = torch.matmul(b, registers)

        # Multiply the output matrix by the arg1, arg2, and e vectors. Also take care
        # of doing the read.
        
        arg1_long = arg1.view(1, -1, 1, 1)
        arg2_long = arg2.view(1, 1, -1, 1)
        instr = e.view(-1, 1, 1, 1)
        read_vec =  e[self.read_index] * torch.matmul(arg1, memory)
        out_vec = (Variable(self.outputs) * arg1_long * arg2_long * instr).sum(0).sum(0).sum(0) + read_vec      
        out_vec = out_vec.squeeze(0)
    
        # Update our memory, registers, instruction register, and stopping probability
        memory = self.writeMemory(e, memory, arg1, arg2)
        registers = self.writeRegisters(out_vec, o, registers)
        IR = self.updateIR(e, IR, arg1, arg2)
        stop_prob = self.getStop(e)
        
        return(memory, registers, IR, stop_prob)
             
    def writeRegisters(self, out, o, registers):
        """
        Write the result of our operation to our registers.
        
        :param out: Probability distribution over the output value (M)
        :param o: Probability distribution over the output register (R)
        :param Registers: register matrix (RxM)
        
        :return: The updated registers (RxM)
        """
        # Multiply probability of not writing with old registers and use an outer product
        return (1 - o).unsqueeze(1) * registers + torch.ger(o, out)
    
    def updateIR(self, e, IR, arg1, arg2):
        """
        Update the instruction register
        
        :param e: Distribution over the current instruction (N)
        :param IR: Instruction register (length M)
        :param arg1: Distribution over the first argument value (length M)
        :param arg2: Distribution over the second argument value (length M)
        
        :return: The updated instruction register (BxMx1)
        """
        # probability of actually jumping
        cond = e[self.jump_index] * arg1[0]
        
        # Take a weighted sum of the instruction register with and without jumping
        return torch.cat([IR[-1], IR[:-1]], 0) * (1 - cond) + arg2 * cond
    
    def writeMemory(self, e, mem_orig, arg1, arg2):
        """
        Update the memory
        
        :param e: Distribution over the current instruction (M)
        :param mem_orig: Current memory matrix (MxM)
        :param arg1: Distribution over the first argument value (M)
        :param arg2: Distribution over the second argument value (M)
        
        :return: The updated memory matrix (MxM)
        """
        
        # Probability that we're on the write instruction
        write_probability = e[self.write_index]
        mem_write = torch.ger(arg1, arg2) 
        mem_write = mem_write + (1 - arg1).unsqueeze(1) * mem_orig
        
        return mem_orig * (1 - write_probability) + write_probability * mem_write

    def getStop(self, e):
        """
        Obtain the probability that we will stop at this timestep based on the probability that we are running the STOP op.
        
        :param e: distribution over the current instruction (length M)
        
        :return: probability representing whether the controller should stop.
        """
        return e[self.stop_index]



