from typing import Any, List, Dict, Optional, Callable, Generator, Tuple, Union

import pandas as pd
import time
import torch
from torch.utils.data import DataLoader
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs
)

from datetime import timedelta
import os
import psutil
import traceback
import numpy as np
from copy import deepcopy

import ray
from ray.air import ScalingConfig
from ray.air import session
from ray.air.result import Result
from ray.train.torch import (
    TorchConfig,
    TorchCheckpoint,
    TorchTrainer as RayTorchTrainer
)

from commons.training_strategy.training_strategy import TrainingStrategy
from commons.base_model_wrapper import BaseModelWrapper, DEFAULT_OPTIM_GROUP
from commons.configs.trainer_pipeline_config import TrainerPipelineConfig
from commons.configs.training_strategy_config import AccelerateTrainingStrategyConfig
from commons.data.data_loader_strategy import DataLoaderStrategy
from commons.pipeline.model_checkpointer import ModelCheckpointer
from commons.pipeline.model_builder import ModelBuilder
from commons.training_strategy.train_loop_per_worker_build import TrainLoopPerWorkerBuilder
from commons.utils import instantiate_class



class AccelerateTrainingStrategy(TrainingStrategy[BaseModelWrapper]):
    def __init__(self, training_strategy_config: AccelerateTrainingStrategyConfig):
        self.training_strategy_config = training_strategy_config

    def train(
        self,
        model_builder: ModelBuilder,
        data_loader_strategy: DataLoaderStrategy,
        train_data_paths: List[str],
        val_data_paths: List[str],
        pipeline_config: TrainerPipelineConfig,
        model_checkpointer: Optional[ModelCheckpointer]
            
    ):
        if pipeline_config.train.skip_train:
            return model_builder.build()
        
        # build per loop train
        train_loop_per_builder = TorchTrainLoopPerWorkerBuilder(
            model_builder,
            data_loader_strategy,
            pipeline_config
        )
        train_loop_per_worker = train_loop_per_builder.build(
            train_data_paths, val_data_paths=val_data_paths,
            model_checkpointer=model_checkpointer
        )

        model_train_config = pipeline_config.train
        resources_per_worker_cfg = None
        if pipeline_config.data_loader.kind.value == "simple":
            resources_per_worker_cfg = {
                "CPU": pipeline_config.data_loader.max_readers + 1,
                "GPU": 1 if model_train_config.use_gpu else 0,
            }

        run_config = ray.air.config.RunConfig()
        run_config.verbose = pipeline_config.log_verbosity
        trainer = RayTorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            run_config=run_config,
            torch_config=TorchConfig(
                timeout_s=model_train_config.distributed_process_group_timeout_s,
            ),
            scaling_config=ScalingConfig(
                num_workers=model_train_config.num_workers,
                use_gpu=model_train_config.use_gpu,
                placement_strategy="SPREAD",
                resources_per_worker=resources_per_worker_cfg,
            ),
        )

        # fit
        result: Result = trainer.fit()

        #build again for inference
        model = model_builder.build()
        # FIXME load the best checkpoint
        ckpt = result.checkpoint
        if hasattr(ckpt, "get_model"):
            model = ckpt.get_model(model=model)
        else:
            with ckpt.as_directory() as checkpoint_dir:
                state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))
                model.load_state_dict(state_dict)
        return model
    

class TorchTrainLoopPerWorkerBuilder(TrainLoopPerWorkerBuilder):
    def __init__(
            self, 
            model_builder: ModelBuilder,
            data_loader_strategy: DataLoaderStrategy,
            pipeline_config: TrainerPipelineConfig
    ):
        super().__init__(model_builder, data_loader_strategy, pipeline_config)
        self.total_batches = 0
        self.num_workers = pipeline_config.train.num_workers
        self.global_batch_size = pipeline_config.train.batch_size * pipeline_config.train.num_workers
        self.checkpoint_every_k_steps = pipeline_config.train.checkpoint_every_k_steps
        self.model_kind = pipeline_config.model.kind
        self.eval_cache = None
        self.extra_day_eval_cache = None

    def per_loop_fit(
            self,
            rank: int,
            per_epoch_dataset_generator: Callable[
                [],
                Generator[Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]], Any, None]
            ],
            config: Optional[Dict],
            model_checkpointer: Optional[ModelCheckpointer] = None,
    ):
        model_train_config = self.pipeline_config.train
        

        # Accelerator reads from this environment variable for GPU placement.
        os.environ["LOCAL_RANK"] = str(session.get_local_rank())
        os.environ["WORLD_SIZE"] = str(session.get_world_size())

        training_strategy_config: AccelerateTrainingStrategyConfig = self.pipeline_config.training_strategy
        gradient_accumulation_steps = self.pipeline_config.train.gradient_accumulation_steps
        kwargs = [
            InitProcessGroupKwargs(timeout=timedelta(seconds=training_strategy_config.timeout)),
            DistributedDataParallelKwargs(
                find_unused_parameters=training_strategy_config.find_unused_parameters,
                static_graph=training_strategy_config.static_graph,
                broadcast_buffers=training_strategy_config.broadcast_buffers,
            ),
        ]
        accelerator = Accelerator(
            device_placement=True,
            cpu=not self.pipeline_config.train.use_gpu,
            # AMP
            mixed_precision=training_strategy_config.precision,
            dispatch_batches=False,
            split_batches=False,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=kwargs,
            **({} if gradient_accumulation_steps is None
               else {'gradient_accumulation_steps': gradient_accumulation_steps})
        )
        model: BaseModelWrapper = self.model_builder.build()
        train_step_fn = model.train_step
        val_step_fn = model.val_step
        param_groups_names = {}
        param_groups = {}

        for mn, m in model.named_modules():
                for pn, p in m.named_parameters():
                    # full param name
                    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L215
                    fpn = '%s.%s' % (mn, pn) if mn else pn
                    optim_group = model.optim_group(m, fpn, p.numel())
                    if optim_group is not None:
                        param_groups_names.setdefault(optim_group, []).append(fpn)

        param_dict = {pn: p for pn, p in model.named_parameters()}

        param_dict = {pn: p for pn, p in model.named_parameters()}

        for gid in param_groups_names:
            param_names = list(sorted(set(param_groups_names[gid])))
            param_groups_names[gid] = param_names
            param_groups[gid] = [param_dict.pop(pn) for pn in param_names]

        if len(param_dict) > 0:
            sentinel_param_group_names = [pn for pn in sorted(param_dict)]
            sentinel_param_group = [param_dict.pop(pn) for pn in sentinel_param_group_names]
            param_groups_names[DEFAULT_OPTIM_GROUP] = sentinel_param_group_names
            param_groups[DEFAULT_OPTIM_GROUP] = sentinel_param_group

        optimizers = model.optimizers_for_param_groups(param_groups)

        if optimizers is None:
            if model_train_config.optimizer_clazz:
                optimizer_kwargs = model_train_config.optimizer_kwargs \
                    if model_train_config.optimizer_kwargs is not None else {}
                optim = instantiate_class(
                    model_train_config.optimizer_clazz, model.parameters(), **optimizer_kwargs)
            else:
                weight_decay = model_train_config.weight_decay \
                    if model_train_config.weight_decay else 0.0
                optim = torch.optim.Adam(
                    model.parameters(),
                    lr=model_train_config.learning_rate,
                    weight_decay=weight_decay,
                )
            optimizer = accelerator.prepare_optimizer(optim)
            if model_train_config.lr_scheduler_clazz:
                scheduler_kwargs = model_train_config.lr_scheduler_kwargs \
                    if model_train_config.lr_scheduler_kwargs is not None else {}
                scheduler_step_size = model_train_config.lr_scheduler_step_size
                scheduler = accelerator.prepare_scheduler(instantiate_class(
                    model_train_config.lr_scheduler_clazz, optimizer, **scheduler_kwargs))
            else:
                scheduler_step_size = 1E9
                scheduler = accelerator.prepare_scheduler(
                    torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
                )
            optimizers = [optimizer]
        else:
            accelerator.print(f'Prepping custom optimizers with {param_groups_names}')
            optimizers = [accelerator.prepare_optimizer(opt) for opt in optimizers]
            scheduler_step_size = 1E9
            scheduler = accelerator.prepare_scheduler(
                torch.optim.lr_scheduler.LambdaLR(optimizers[0], lr_lambda=lambda _: 1.0)
            )

        print(f"Worker# {rank} - Starting train loop on worker")
        try:
            current_epoch = 0
            for train_dl, val_dl, extra_day_val_dl in per_epoch_dataset_generator():
                # train_dl = accelerator.prepare_data_loader(train_dl)
                if self.eval_cache is None:
                    self.init_eval_cache(accelerator, val_dl)
                if self.extra_day_eval_cache is None:
                    self.init_eval_cache(accelerator, extra_day_val_dl, for_extra_day=True)

                print(f"Worker# {rank} - Reading epoch: #{current_epoch}")
                metrics = self.train_epoch(
                    accelerator,
                    train_dl,
                    model,
                    optimizers,
                    scheduler,
                    scheduler_step_size,
                    rank,
                    current_epoch,
                    model_train_config.epochs,
                    train_step_fn=train_step_fn,
                    val_step_fn=val_step_fn,
                    model_checkpointer=model_checkpointer if rank == 0 and self.checkpoint_every_k_steps is not None
                    else None,
                )

                # all workers need to checkpoint
                state_dict = accelerator.get_state_dict(model, unwrap=True)
                session.report(
                    metrics=metrics,
                    checkpoint=TorchCheckpoint.from_state_dict(
                        state_dict
                    ),
                )
                current_epoch += 1
            # mlflow finished status will be set in parent process
            self.pipeline_config.trackers.end_run()
        except Exception as e:
            print(f"Error in worker#{rank}", e, traceback.format_exc())
            if rank == 0:
                self.pipeline_config.trackers.end_run(error=True)
            raise e

         
    def init_eval_cache(self, accelerator: Accelerator, dataloader: DataLoader) -> None:
        if self.pipeline_config.train.validation_steps > 0:
            self.eval_cache = []
            for idx, val_batch in enumerate(dataloader):
                if idx % self.pipeline_config.train.cache_every_k_val_batch == 0:
                    print(f"caching batch number {idx}")

                batch = {
                    f: val_batch[f] if isinstance(val_batch[f], torch.Tensor) else val_batch[f]
                    for f in val_batch
                }
                self.eval_cache.append(batch)
                
                if (len(self.eval_cache) >= self.pipeline_config.train.validation_steps):
                    break

                  
    def train_epoch(
        self,
        accelerator: Accelerator,
        train_dataloader: torch.utils.data.DataLoader,
        model: Union[BaseModelWrapper, torch.nn.parallel.DistributedDataParallel],
        optimizers: List[torch.optim.Optimizer],
        scheduler,
        scheduler_step_size,
        rank: int,
        current_epoch: int,
        epochs: int,
        train_step_fn: Callable,
        val_step_fn: Callable,
        model_checkpointer: Optional[ModelCheckpointer] = None,
    ) -> Dict[str, float]:
        train_log_steps = self.pipeline_config.train.train_metrics_every_n_steps
        val_log_steps = self.pipeline_config.train.val_metrics_every_n_steps

        metrics_agg = {}
        metrics_agg_num = 0
        batch_nb = 0
        dataloader_iter = iter(train_dataloader)
        device = accelerator.device
        best_loss = float('inf')
        best_model_after_k_steps = self.pipeline_config.export.best_model_after_k_steps if \
            self.pipeline_config.export.best_model_after_k_steps else 0
        loss_factor_for_exporting = self.pipeline_config.export.export_if_loss_within_factor_of_best_model if \
            self.pipeline_config.export.export_if_loss_within_factor_of_best_model else float('inf')
        global_num_samples = 0
        global_metrics = {}
        global_extra_day_val_metrics = {}
        while True:
            model.train()
            try:
                batch = next(dataloader_iter)
                # check if others have told us to stop training
                if self.do_we_need_to_stop_training_syncer(False, device):
                    break
            except StopIteration:
                # let others know that we need to stop training
                _ = self.do_we_need_to_stop_training_syncer(True, device)
                break

            batch_size = next(iter(batch.values())).shape[0] * self.num_workers

            batch = {
                key: batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
                for key in batch
            }

            if self.total_batches == 0:
                self.train_start_time = time.time()
                if rank == 0:
                    self.pipeline_config.trackers.watch(accelerator.unwrap_model(model), log_graph=True)

            self.total_batches += 1

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    output = model(batch)
                    loss, metrics = train_step_fn(batch, output)
                    loss = loss + 0.0 * sum(param.abs().sum() for param in model.parameters())
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        model_train_config = self.pipeline_config.train
                        if model_train_config.gradient_clip_norm:
                            accelerator.clip_grad_norm_(model.parameters(), model_train_config.gradient_clip_norm)
                        if model_train_config.gradient_clip_value:
                            accelerator.clip_grad_value_(model.parameters(), model_train_config.gradient_clip_value)
                    for optim in optimizers:
                        optim.step()
                    if self.total_batches % scheduler_step_size == (scheduler_step_size - 1):
                        scheduler.step()
                    for optim in optimizers:
                        optim.zero_grad()

            if batch_nb % train_log_steps == 0:
                print(f'TRAIN EPOCH: {current_epoch + 1}/{epochs}: BATCH: {batch_nb} RANK: {rank} '
                      f'LOSS: {loss} METRICS: {metrics}')

            if rank == 0 and model_checkpointer is not None and (batch_nb + 1) % self.checkpoint_every_k_steps == 0:
                reasons = []
                stop = False
                dont_export = False
                if any(torch.isnan(param).any() for param in model.parameters()):
                    reasons.append("NaNs in model parameters")
                    stop = True
                    dont_export = True
                if torch.isnan(loss):
                    reasons.append("NaN in train loss")
                    stop = True
                    dont_export = True
                if best_loss > 0.0 and loss.cpu().item() > loss_factor_for_exporting * best_loss:
                    dont_export = True
                if not dont_export:
                    print(f"CHECKPOINTING FOR BATCH {batch_nb}!!!")
                    module_ = accelerator.unwrap_model(model)
                    model_checkpointer.checkpoint(state_dict=deepcopy(module_).state_dict(),
                                                  result_df=pd.DataFrame({k: [v] for k, v in global_metrics.items()}),
                                                  result_extra_day_df=pd.DataFrame({k: [v] for k, v in global_extra_day_val_metrics.items()}))
                else:
                    print(f"SKIPPING CHECKPOINTING FOR BATCH {batch_nb}! CURRENT: {loss.cpu().item()}, "
                          f"BEST: {best_loss}!!!")
                if stop:
                    raise ValueError(f"Stopping because of {','.join(reasons)}!")
            if rank == 0 and model_checkpointer is not None and len(global_metrics) > 0 and \
                    batch_nb % train_log_steps == 0:
                model_checkpointer.checkpoint(state_dict=None,
                                              result_df=pd.DataFrame({k: [v] for k, v in global_metrics.items()}),
                                              result_extra_day_df=pd.DataFrame({k: [v] for k, v in global_extra_day_val_metrics.items()}))

            if len(metrics_agg.keys()) == 0:
                metrics_agg = metrics
            else:
                for name in metrics:
                    metrics_agg[name] += metrics[name]
            metrics_agg_num += 1

            # global_num_samples = self.global_batch_size * self.total_batches
            global_num_samples += batch_size

            if rank == 0 and batch_nb % train_log_steps == 0:
                for name in metrics:
                    metrics_agg[name] = metrics_agg[name] / metrics_agg_num
                training_speed = global_num_samples / (time.time() - self.train_start_time)
                report_metrics = {"training speed - samples per second": training_speed,
                                  "epoch": current_epoch, "steps": batch_nb}
                for i, lr in enumerate(scheduler.get_last_lr()):
                    metrics_agg[f'learning_rate_for_group_{i}'] = lr

                self.pipeline_config.trackers.log_metrics(metrics_agg, step=global_num_samples)

                metrics_agg = {}
                metrics_agg_num = 0

                self.pipeline_config.trackers.log_metrics(report_metrics,
                                                          step=global_num_samples)
                accelerator.print(f"Training speed: {training_speed} samples/s")
                global_metrics.update(metrics)
                global_metrics['train_speed'] = training_speed
                global_metrics['train_epoch'] = current_epoch
                global_metrics['train_steps'] = global_num_samples
                global_extra_day_val_metrics.update(metrics)
                global_extra_day_val_metrics['train_speed'] = training_speed
                global_extra_day_val_metrics['train_epoch'] = current_epoch
                global_extra_day_val_metrics['train_steps'] = global_num_samples
                

            if batch_nb % val_log_steps == 0:
                val_loss, val_metrics = self.val(model=model,
                                                 step=global_num_samples,
                                                 accelerator=accelerator,
                                                 val_step_fn=val_step_fn)
                
                print(f'VAL EPOCH: {current_epoch + 1}/{epochs}: BATCH: {batch_nb} RANK: {rank} '
                      f'LOSS: {val_loss} METRICS: {val_metrics}')
                if rank == 0:
                    self.pipeline_config.trackers.log_metrics(val_metrics, step=global_num_samples)
                    global_metrics.update(val_metrics)
                    global_metrics['val_epoch'] = current_epoch
                    global_metrics['val_steps'] = global_num_samples
                    global_extra_day_val_metrics['val_epoch'] = current_epoch
                    global_extra_day_val_metrics['val_steps'] = global_num_samples
            if self.checkpoint_every_k_steps is not None and batch_nb >= best_model_after_k_steps and \
                    (batch_nb + 1) % self.checkpoint_every_k_steps == 0:
                best_loss = min(best_loss, loss.cpu().item())
            batch_nb += 1
        accelerator.print(f'rank {rank} epoch finished with {batch_nb} steps')
        return metrics_agg

    @staticmethod
    def do_we_need_to_stop_training_syncer(no_data_on_this_rank, current_device):
        no_data_on_this_rank_tensor = torch.tensor(no_data_on_this_rank, device=current_device).to(torch.float32)
        # create a buffer
        tensors_gather = [torch.ones_like(no_data_on_this_rank_tensor) for _ in
                          range(torch.distributed.get_world_size())]

        # get tensors from all ranks
        try:
            torch.distributed.all_gather(tensors_gather, no_data_on_this_rank_tensor, async_op=False)
        except Exception as e:
            print(f"Caught Exception {str(e)}, traceback: \n{traceback.format_exc()}")
            exit()
        # check if any of the flags is not zero
        if torch.sum(torch.stack(tensors_gather, dim=0)) > 0:
            return True
        return False

    def val(
            self,
            model,
            step: int,
            accelerator: Accelerator,
            val_step_fn: Callable,
            for_extra: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        model.eval()
        val_loss = 0
        num_batches = 0
        metrics_agg = {}
        t0 = time.time()
        batches_skipped = 0
        with torch.no_grad():
            with accelerator.no_sync(model):
                cache = self.extra_day_eval_cache if for_extra else self.eval_cache
                for val_batch in cache:
                    batch = {
                        f: val_batch[f].to(accelerator.device)
                        if isinstance(val_batch[f], torch.Tensor) else val_batch[f]
                        for f in val_batch
                    }

                    output = model(batch)
                    loss, metrics = val_step_fn(batch, output)
                    val_loss += loss.item()
                    found_nan = any([np.isnan(metrics[name]) for name in metrics])
                    if found_nan:
                        batches_skipped += 1
                        # if we found nan, we skip this batch for all metrics
                        continue

                    for name in metrics:
                        metrics_agg[name] = metrics_agg.setdefault(name, 0) + metrics[name]
                    num_batches += 1

        print("Skipped val batches due to NAN metrics: ", batches_skipped)
        eval_speed = len(cache) * self.pipeline_config.eval.eval_batch_size * \
                     self.pipeline_config.train.num_workers / (time.time() - t0)
        log_dict = {"eval speed - samples per second": eval_speed,
                    'RAM Available - GB': psutil.virtual_memory().available / 1000000000}
        # global_num_samples = self.global_batch_size * self.total_batches
        self.pipeline_config.trackers.log_metrics(log_dict, step=step)

        # Properly normalize everything - need to exchange num_batches in this case
        val_metrics = aggregate_dict(metrics_agg, num_batches=num_batches, device=accelerator.device)
        return val_loss, val_metrics


def aggregate_dict(input: Dict[str, float], num_batches: int, device: torch.device) -> Dict[str, float]:
    output: Dict[str, float] = {}
    # Loop over the entries and aggregate, but get the number of batches first (since it may vary per rank)
    agg_num_batches = aggregate_tensor(torch.tensor([float(num_batches)]).to(device))
    # NOTE: Just in case, exchanging tensors in a consistent way by sorting the keys first
    for m in sorted(input.keys()):
        output[m] = aggregate_tensor(torch.tensor([input[m]]).to(device)) / agg_num_batches
    return output


def aggregate_tensor(tensor: torch.Tensor) -> float:
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    summed_tensor = torch.sum(torch.stack(tensors_gather, dim=0))
    return summed_tensor.item()