from typing import Dict, Tuple, List, Optional, Any

import torch
import torch.nn as nn

DEFAULT_OPTIM_GROUP = 'DEFAULT_OPTIM_GROUP'


class BaseModelWrapper(nn.Module):
    def __init__(self, dummy_params: bool = False, sparse: bool = False):
        super().__init__()
        if dummy_params:
            if sparse:
                # Add a dummy dense parameter to keep the dense optimizer occupied
                self.dummy_dense_emb = nn.Embedding(1, 1)
            else:
                # Add a dummy sparse parameter to keep the sparse optimizer occupied
                self.dummy_sparse_emb = nn.Embedding(1, 1, sparse=True)

    def train_step(self, batch: Dict[str, torch.Tensor], output: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError('Subclasses must implement this method')

    def val_step(self, batch: Dict[str, torch.Tensor], output: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError('Subclasses must implement this method')

    def is_sparse(self, param_name: str):
        return param_name == 'dummy_sparse_emb'

    def inference_models(self, batch: Optional[Any] = None) -> List[torch.jit.ScriptModule]:
        raise NotImplementedError('Subclasses must implement this method')

    # methods below are for torch parameter server
    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        self.load_state_dict(weights)

    def get_gradients(self) -> List[Optional[torch.Tensor]]:
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients: List[Optional[torch.Tensor]]) -> None:
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = g

    def optim_group(self, parent_module: nn.Module, full_param_name: str, numel: int) -> Optional[str]:
        """
        Assign optim group to parameters. Please see all-reduce for how this method is called

        :param parent_module: one of the possible parent modules for the parameter (one looks for whether this is an
            nn.Embedding or nn.Linear, etc)
        :param full_param_name: fully qualified parameter name (generally one looks for whether this ends in a specific
            pattern like ".bias", ".weight", etc)
        :param numel: how big is this parameter (number of elements)?
        :return: parameter group name
        """
        return None

    def optimizers_for_param_groups(self, param_groups: Dict[str, List[torch.nn.Parameter]]) -> \
            Optional[List[torch.optim.Optimizer]]:
        """

        :param param_groups: dict with parameters assigned to optim groups specified above. any params that are not
            assigned to any optim group will be assigned a special optim group: "DEFAULT_OPTIM_GROUP"
        :return:
        """
        return None