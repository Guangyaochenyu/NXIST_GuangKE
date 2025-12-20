import os
import torch
import torch.nn as nn

__all__ = ['BasicModule']

class BasicModule(nn.Module):

    def __init__(self, processor, config):
        super(BasicModule, self).__init__()
        self.processor  = processor
        self.config     = config

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
        return self

    def save(self, epoch):
        prefix = os.path.join(
            os.environ.get('KE_HOME'), 
            'checkpoints', 
            f"{os.environ.get('KE_MODEL')}-{os.environ.get('KE_DATA')}",
            os.environ.get('KE_TIME'),
        )
        os.makedirs(prefix, exist_ok=True)
        name = os.path.join(prefix, f"epoch_{epoch:03d}.pth")
        torch.save(self.state_dict(), name)
        return name
