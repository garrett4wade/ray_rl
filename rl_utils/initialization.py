import torch.nn as nn


def init(m):
    if isinstance(m, nn.Sequential):
        for c in m.children():
            init(c)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
    elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.ReLU):
        pass
    else:
        raise NotImplementedError
