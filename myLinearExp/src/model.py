import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticModel(torch.nn.Module):

    def __init__(self, input_featrue, output_feature=1):
        super(LogisticModel, self).__init__()
        self.input_featrue = input_featrue
        self.fc = nn.Linear(input_featrue, output_feature, bias=False)
        
    def forward(self, x):
        y_pred = self.fc(x.view(-1, self.input_featrue))
        y_pred = torch.sigmoid(y_pred)
        return y_pred