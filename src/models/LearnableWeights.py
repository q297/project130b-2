import torch.nn as nn
import torch

class LearnableWeights(nn.Module):
    def __init__(self, n_models):
        super().__init__()
        self.raw_w = nn.Parameter(torch.zeros(n_models))

    def forward(self, model_probs):
        w = torch.softmax(self.raw_w, dim=0)
        return (model_probs * w).sum(dim=1)
