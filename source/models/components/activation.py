import torch.nn as nn


class Activation(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act
        
    def forward(self, x):
        return self.act(x)


class ReLU(Activation):
    def __init__(self, inplace=False):
        super().__init__(nn.ReLU(inplace=inplace))


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(nn.Sigmoid())
    
    
class Tanh(Activation):
    def __init__(self):
        super().__init__(nn.Tanh())