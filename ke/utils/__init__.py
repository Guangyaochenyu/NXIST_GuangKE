from .ioUtils import *
from .processor import *

def resolve_to_environ(config):
    import os
    os.environ['KE_MODEL'] = config.model.name
    os.environ['KE_DATA']  = config.data.name