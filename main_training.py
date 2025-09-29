import hydra
import pydantic
import ray
import logging
from omegaconf import DictConfig, OmegaConf

from commons.configs import model_config
from commons.configs.model_config import ModelKind
from commons.configs.trainer_pipeline_config import TrainerPipelineConfig, init_hydra
from commons.data import get_data_loader_strategy
from commons.training_strategy import get_training_strategy
from commons.pipeline.trainer_pipeline import TrainerPipeline

from models.ranker.config import RankerModelConfig
from models.lthm.config import LTHMModelConfig



init_hydra()

def execute_pipeline(pipeline_cfg: TrainerPipelineConfig):
    logging.info("Start execute pipeline")
    logging.basicConfig(
        format='%(asctime)s, %(levelname)-1s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:::%H:%M:%S',
        level=logging.INFO)
    
    # model builder
    data_mapper_fn = pipeline_cfg.model.preprocess_fn
    if  pipeline_cfg.model.kind == ModelKind.RANKER and isinstance(pipeline_cfg.model, RankerModelConfig):
            model_builder = pipeline_cfg.model.get_builder(stats=None)

    elif  pipeline_cfg.model.kind == ModelKind.LTHM and isinstance(pipeline_cfg.model, LTHMModelConfig):
            model_builder = pipeline_cfg.model.get_builder(stats=None)

    # data loader strategy
    data_loader_strategy = get_data_loader_strategy(
        pipeline_cfg.data_loader,
        columns=pipeline_cfg.model.features.get_input_columns(),
        data_mapper=data_mapper_fn
    )

    #get training strategy
    training_strategy = get_training_strategy(pipeline_cfg.training_strategy)

    #get trainer pipeline
    pipeline_builder = TrainerPipeline
    pipeline = pipeline_builder(
            pipeline_config=pipeline_cfg,
            model_builder=model_builder,
            training_strategy=training_strategy,
            data_loader_strategy=data_loader_strategy,
        )
    pipeline.execute()


@hydra.main(version_base=None, config_path="hydra-configs")
def main_app(cfg: DictConfig) -> None:
    try:
        OmegaConf.resolve(cfg)
        cfg_dict = OmegaConf.to_object(cfg)
        print("Using resolved config: \n", OmegaConf.to_yaml(cfg_dict))
        cfg_dict["config_str"] = OmegaConf.to_yaml(cfg_dict)
        print(model_config.model_registry)

        pipeline_cfg = TrainerPipelineConfig.model_validate(cfg_dict)

        print("Training Strategy Class: ", pipeline_cfg.training_strategy.__class__)
        print("Model Class: ", pipeline_cfg.model.__class__)
        print("Model Version: ", pipeline_cfg.model_version)
        print("Model Run Id: ", pipeline_cfg.run_id)
        print("Trackers Project: ", pipeline_cfg.trackers.project_name)
        print("Trackers Experiment Name: ", pipeline_cfg.trackers.experiment_name)
        print("Trackers Run Name: ", pipeline_cfg.trackers.run_name)


        try:
            context = ray.init()
            logging.info(f"Ray initialized: {context}")
            execute_pipeline(pipeline_cfg)
        finally:
            ray.shutdown()



    except pydantic.ValidationError as e:
        print(e.errors())
        raise RuntimeError("Error in validating configs", e)
    

if __name__ == "__main__":
    main_app()