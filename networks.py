import torch
import torch.nn.functional as F
import torch.nn as nn

class DATVIL(nn.Module):
    def __init__(self, attributes, alpha, dtype, plus_residual = 1, plus_transform= 1):
        super().__init__()
        
        self.attributes = attributes

        self.alpha = alpha

        self.dtype = dtype
        
        self.plus_residual = plus_residual
        self.plus_transform = plus_transform
        
        if self.plus_residual:
            self.text_attributes_residuals = nn.Parameter(torch.zeros(attributes.size()).to(self.dtype).cuda())
        if self.plus_transform:
            self.text_attributes_weights = nn.Parameter(torch.eye(attributes.size(1)).to(self.dtype).cuda())

    def forward(self):
        if self.plus_residual and self.plus_transform:
            return self.attributes @ self.text_attributes_weights + self.alpha * self.text_attributes_residuals
        elif self.plus_residual:
            return self.attributes + self.alpha * self.text_attributes_residuals
        elif self. plus_transform:
            return self.attributes @ self.text_attributes_weights
        