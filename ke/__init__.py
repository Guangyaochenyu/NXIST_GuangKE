import os
import sys
import time
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('Guang-KE')
logger.setLevel(logging.INFO)

ke_home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(ke_home)
os.environ['KE_HOME'] = ke_home
os.environ['KE_MODEL'] = 'Default'
os.environ['KE_DATA'] = 'Default'
os.environ['KE_TIME'] = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from . import utils
from . import models

__all__ = ['models', 'utils', 'logger']