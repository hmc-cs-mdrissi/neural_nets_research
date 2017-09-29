import torch
import torch.nn as nn
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class ResidualBlock(nn.Module):
    def __init__(self, planes):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class ResidualNet(nn.Module):
    def __init__(self, input_planes, height, width, number_of_blocks, classes):
        super(ResidualNet, self).__init__()

        if number_of_blocks < 2:
            raise ValueError("The residual net needs at least two blocks.")

        self.conv1 = conv3x3(input_planes, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.residual1 = ResidualBlock(16)
        self.conv2 = conv3x3(16, 32, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.residual2 = ResidualBlock(32)
        self.conv3 = conv3x3(32, 64, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.later_residual_blocks = nn.ModuleList()

        for _ in range(number_of_blocks-2):
            self.laterResidualBlocks.append(ResidualBlock(64))

        self.dense_input_dim = height * width * 4
        self.dense = nn.Linear(self.dense_input_dim, classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.residual1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.residual2(x)
        x = self.relu(self.bn3(self.conv3(x)))

        for block in self.later_residual_blocks:
            x = block(x)

        return self.dense(x.view(-1, self.dense_input_dim))

class MyLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyLSTMLayer, self).__init__()

        self.hidden_size = hidden_size

        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.incorporate_position_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.incorporate_value_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_value_gate = nn.Linear(hidden_size, hidden_size)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward_step(self, input, hidden, cell_state):
        combined = torch.cat((input, hidden), 1)

        f = self.sigmoid(self.forget_gate(combined))
        i = self.sigmoid(self.incorporate_position_gate(combined))
        C_new = self.tanh(self.incorporate_value_gate(combined))

        cell_state = f * cell_state + i * C_new

        hidden = self.relu(self.hidden_value_gate(cell_state))

        return hidden, cell_state

    def forward(self, input):
        hidden, cell_state = self.initAll()

        outputs = []

        for i in range(input.size()[0]):
            hidden, cell_state = self.forward_step(input[i], hidden, cell_state)
            outputs.append(hidden)

        return torch.stack(outputs)

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size)).cuda()

    def init_cell_state(self):
        return Variable(torch.zeros(1, self.hidden_size)).cuda()

    def init_all(self):
        return self.init_hidden(), self.init_cell_state()

class MyLSTM(nn.Module):
    def __init__(self, net_input_size, hidden_sizes, output_size, layers):
        super(MyLSTM, self).__init__()

        self.lstm_layers = nn.ModuleList()
        input_sizes = [net_input_size] + hidden_sizes[:-1]

        if len(hidden_sizes) != layers:
            raise ValueError("The number of layers should match the number of hidden sizes given.")

        for input_size, hidden_size in zip(input_sizes, hidden_sizes):
            self.lstm_layers.append(MyLSTMLayer(input_size, hidden_size))

        self.outputGate = nn.Linear(hidden_sizes[-1], output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        hiddens = input

        for lstm_layer in self.lstm_layers:
            hiddens = lstm_layer(hiddens)

        return self.softmax(self.outputGate(hiddens[-1]))
