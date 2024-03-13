import torch.nn as nn
import torch.nn.functional as F

class HandMLP(nn.Module):
    def __init__(self, output_size=3):
        super().__init__()
        self.input_size = 63
        self.hidden1_size = 64
        self.hidden2_size = 64
        self.hidden3_size = 32
        self.output_size = output_size

        self.layer1 = nn.Linear(self.input_size, self.hidden1_size)
        self.layer2 = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.layer3 = nn.Linear(self.hidden2_size, self.hidden3_size)
        self.output = nn.Linear(self.hidden3_size, self.output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.output(x)
        return x