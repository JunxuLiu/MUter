import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(torch.nn.Module):

    def __init__(self, input_featrue, output_feature=1):
        super(LinearModel, self).__init__()
        self.input_featrue = input_featrue
        self.fc = nn.Linear(input_featrue, output_feature, bias=False)
        
    def forward(self, x):
        return self.fc(x.view(-1, self.input_featrue))