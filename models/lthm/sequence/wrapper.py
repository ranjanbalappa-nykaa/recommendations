from typing import Dict, Optional, List
from collections import OrderedDict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from commons.layers import CascadedStreamingLogQCorrectionModule
from commons.pipeline.types import Stats
from commons.base_model_wrapper import BaseModelWrapper
from models.lthm.sequence.encoder import Encoder
from models.lthm.config import LTHMModelConfig


class LTHMModelWrapper(BaseModelWrapper):
    def __init__(self, model_config: LTHMModelConfig, stats: Stats):
        super().__init__(dummy_params=model_config.sparse, sparse=model_config.sparse)
        self.model_config = model_config
        self._sparse = model_config.sparse
        self._tasks = model_config.tasks
        self._softmax_temperature = model_config.softmax_temperature
        self._export_span = model_config.export_span
        self._export_tokens = model_config.export_tokens
        self._loss_type = model_config.loss_type
        self._metrics_k_all = model_config.metrics_k_all
        self._lookahead = model_config.lookahead
        self.features = model_config.features
        

       
        #Encoder
        self._model = Encoder(model_config)


        #LogqCorrection
        log_q_config = model_config.log_q_config
        self._log_q_beta = log_q_config.beta
        self._log_q_calc = CascadedStreamingLogQCorrectionModule(
            num_buckets=log_q_config.num_buckets,
            hash_offsets=log_q_config.hash_offsets,
            alpha=log_q_config.alpha,
            p_init=log_q_config.p_init,
        )
        self.batch_idx = 0


    def format_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        #format the input data formats
        for f in self.features.categorical_history_features:
            f_name = f.name
            assert batch[f_name].dtype == torch.int64, f"{f_name} was expected to be of type long but was " \
                                                       f"{batch[f_name].dtype}"
            
        for f in self.features.timestamp_features:
            f_name = f.name
            assert batch[f_name].dtype == torch.int64, f"{f_name} was expected to be of type long but was " \
                                                       f"{batch[f_name].dtype}"
            
        for f in self.features.tensor_list_features:
            f_name = f.name
            batch[f_name] = batch[f_name].float()

        return batch

        
    def forward(self, batch: Dict[str, torch.Tensor]):
       batch =  self.format_inputs(batch)
       return self._model(batch)

    
    def train_step(self, batch, output):
        return self._mini_batch_mapper(batch, output, True)
    
    def val_step(self, batch, output):
        return self._mini_batch_mapper(batch, output)
    
    def _mini_batch_mapper(self, batch, output, training: bool = False):
        if not training:
            return self._train_or_val_step_helper(batch, output, False)
        
        if self._model_config.train_mini_batch_size < 0:
            return self._train_or_val_step_helper(batch, output, True)
        
        #constructs min batch
        batch_size = output['next_token_emb'].size(0)
        mini_batch_size = self._model_config.train_mini_batch_size
        num_mini_batches = batch_size // mini_batch_size
        if num_mini_batches * mini_batch_size < batch_size:
            num_mini_batches += 1

        loss = 0
        metrics = {}
        n_mini_batches = {}
        for i in range(num_mini_batches):
            batch_ = {
                key: batch[key][(i * mini_batch_size) : min((i+1) * mini_batch_size, batch_size)] for key in batch
            }
            output_ = {
                key: output[key][(i * mini_batch_size):min((i + 1) * mini_batch_size, batch_size)] for key in output
            }
            loss_, metrics_ = self._train_or_val_step_helper(batch_, output_, True)
            loss = loss + loss_ / num_mini_batches
            for key in metrics_:
                n_mini_batches[key] = n_mini_batches.setdefault(key, 0) + 1
                metrics[key] = metrics.setdefault(key, 0) + metrics_[key]

        for key in metrics:
            metrics[key] = metrics[key] / n_mini_batches[key]

        metrics['train_overall_batch_size'] = int(batch_size)
        return loss, metrics
    
    def _train_or_val_step_helper(self, batch, output, training: bool):
        """
        Loss & metrics calc helper method
        """
        output_emb = F.normalize(output['next_token_emb'], p=2.0, dim=-1)
        input_emb = F.normalize(output['current_token_emb'], p=2.0, dim=-1)
        mask = output['current_token_mask']

        device = output_emb.device
        batch_size = output_emb.size(0)

        emb_dim = output_emb.size(-1)
        seq_len = input_emb.size(-2)
        assert input_emb.size(-1) == emb_dim, f"{input_emb.size(-1)} != {emb_dim}"
        assert output_emb.size(1) == seq_len + 1
        assert output_emb.size(2) == self._export_tokens

        #logq correction on ids
        product_ids = output['current_token_id']
        product_id_flattened = product_ids.view(-1)[mask.view(-1) == 0]
        self._log_q_calc.train_step(product_id_flattened, self.batch_idx) #train logqcorrection
        log_q_correction = self._log_q_calc(product_ids)
        self.batch_idx += 1

        step_type = 'train' if training else 'val'
        metrics = {
            f'{step_type}_batch_size': batch_size,
            f'{step_type}_seq_len': seq_len,
        }
        loss = torch.zeros((1,), device=device)

        #iterate for each lookahead
        previous_offset = 0
        for i, max_offset in enumerate(self._lookahead):
            if i == 0:
                offset = max_offset
                previous_offset = offset
            else:
                offset = random.randint(previous_offset + 1, max_offset)
                previous_offset = offset

            mask_ = mask[:, offset:].contiguous()
            log_q_correction_ = log_q_correction[:, offset:].contiguous()
            this_seq_len = seq_len - offset
            if this_seq_len <= 0:
                continue

            input_emb_ = input_emb[:, offset:].reshape(-1, emb_dim)
            output_emb_ = output_emb[:, :this_seq_len, i].reshape(-1, emb_dim)
            bs_ = output_emb_.size(0)
            labels = torch.arange(0, bs_, device=device)

            #zero out logqcorrection for positive
            log_q_correction_ = log_q_correction_.reshape(1, -1).repeat(bs_, 1)
            log_q_correction_.index_put_(
                (labels, labels),
                torch.zeros_like(labels, dtype=log_q_correction_.dtype),
                accumulate=False,
            )
            

            pos = (
                torch.arange(0, batch_size, dtype=torch.long, device=device).unsqueeze(1)
                .repeat(1, this_seq_len)
                .view(-1, 1)
            )
            pos_matrix = torch.eq(pos, pos.T)
            eye = torch.eye(bs_, dtype=torch.bool, device=device)
            mask_ = mask_.view(-1) 

            #calculate logits
            logits = (output_emb_ @ input_emb_.T) / self._softmax_temperature

            #apply mask and other things
            logits = torch.where(pos_matrix & ~eye, -float('inf'), logits)
            logits = torch.where(mask_.unsqueeze(0), -float('inf'), logits)
            logits = torch.where(mask_.unsqueeze(1), -float('inf'), logits)

            #dont do anything if you dont have negatives because it has only padding
            num_negatives = (~torch.isinf(logits)).sum(dim=-1) - 1
            not_use = torch.logical_or(mask_, num_negatives <= 0)
            if bool(not_use.all().cpu().numpy()):
                continue

            logits = logits[~not_use]
            labels = labels[~not_use]
            num_negatives = num_negatives[~not_use]
            log_q_correction_ = log_q_correction_[~not_use]

            #calculate loss
            loss_all_tokens_unreduced = F.cross_entropy(
                (logits - self._log_q_beta * log_q_correction_) if log_q_correction_ is not None else logits,
                labels,
                reduction='none'
            )

            #filter nan in loss
            loss_all_tokens_unreduced = loss_all_tokens_unreduced[~loss_all_tokens_unreduced.isnan()]
            used_tokens = loss_all_tokens_unreduced.size(0)
            if used_tokens == 0:
                continue
            
            #update loss
            loss_all_tokens = loss_all_tokens_unreduced.mean()
            loss = loss + loss_all_tokens

            #update metrics
            metrics.update({
                f'{step_type}_effective_batch_size_offset_{offset}': logits.size(0),
                f'{step_type}_average_negatives_per_token_offset_{offset}': num_negatives.mean(dtype=torch.float),
                f'{step_type}_used_tokens_offset_{offset}': used_tokens,
                f'{step_type}_loss_all_tokens_offset_{offset}': loss_all_tokens.detach(),
            })

            _, pos = torch.nonzero(
                torch.argsort(logits, dim=-1, descending=True) == labels.unsqueeze(-1),
                as_tuple=True
            )
            metrics[f'{step_type}_average_hit_position_offset_{offset}'] = pos.mean(dtype=torch.float)
            metrics[f'{step_type}_median_hit_position_offset_{offset}'] = pos.float().quantile(q=0.5)

            for k_ in self._metrics_k_all:
                k = min(k_, int(num_negatives.min().cpu().numpy()))
                metrics[f'{step_type}_hit_rate_at_{k_}_offset_{offset}'] = \
                    (logits.topk(k=k)[1] == labels.unsqueeze(-1)).sum(dim=1).mean(dtype=torch.float).detach()

        metrics.update({
            f'{step_type}_loss': loss.detach(),
        })

        self._convert_metrics_tensor_to_float(metrics)
        return loss, metrics


    @staticmethod
    def _convert_metrics_tensor_to_float(metrics):
        for key in metrics:
            if isinstance(metrics[key], torch.Tensor):
                metrics[key] = float(metrics[key].cpu().numpy())


    def is_sparse(self, param_name: str):
        if super().is_sparse(param_name):
            return True
        return False

    def optim_group(self, parent_module: nn.Module, full_param_name: str, numel: int) -> Optional[str]:
        return 'USE_OPTIM'

    def optimizers_for_param_groups(self, param_groups: Dict[str, List[torch.nn.Parameter]]) -> \
            Optional[List[torch.optim.Optimizer]]:
        optim_clazz = torch.optim.AdamW
        result = []
        if len(param_groups.setdefault('USE_OPTIM', [])) > 0:
            optim_ = optim_clazz(
                param_groups['USE_OPTIM'],
                lr=self._model_config.lr,
                weight_decay=self._model_config.weight_decay,
                betas=self._model_config.betas
            )
            result.append(optim_)
        return result
