import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as Tensor

from models.ranker.config import RankingModelConfig
from models.ranker.fdlrm.towers.query import QueryTower
from models.ranker.fdlrm.towers.product import ProductTower

        
        
