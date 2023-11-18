import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_gram_matrix(x):
    """
    Takes a matrix as input. Multiplies it with its transpose to generated a Gram Matrix
    """
    
    a, b, c, d = x.size()
    features = x.view(a*b, c*d)
    G = torch.mm(features, features.t())
    return G.div(a*b*c*d)


class ContentLoss(nn.Module):
    def __init__(self, target) -> None:
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    
    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    def __init__(self, target) -> None:
        super(StyleLoss, self).__init__()
        self.target = _get_gram_matrix(target).detach()
    
    def forward(self, x):
        G = _get_gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x
    

class Normalization(nn.Module):
    def __init__(self, mean, std) -> None:
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)
    
    def forward(self, x):
        return (x - self.mean) / self.std