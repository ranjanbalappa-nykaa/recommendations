from commons.configs.training_strategy_config import TrainingStrategyConfig, OneGpuTrainingStrategyConfig
from commons.training_strategy.accelerate_training_strategy import AccelerateTrainingStrategy
from commons.configs.training_strategy_config import AccelerateTrainingStrategyConfig


def get_training_strategy(training_strategy_config: TrainingStrategyConfig) -> TrainingStrategyConfig:
    if isinstance(training_strategy_config, AccelerateTrainingStrategyConfig):
        training_strategy = AccelerateTrainingStrategy(training_strategy_config)

        return training_strategy
    
    raise ValueError(f"Not supported training strategy: {training_strategy_config}")