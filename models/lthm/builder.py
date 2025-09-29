from commons.pipeline.model_builder import ModelBuilder
from commons.base_model_wrapper import BaseModelWrapper
from commons.pipeline.types import Stats
from models.lthm.sequence.wrapper import LTHMModelWrapper
from models.lthm.config import LTHMModelConfig


class LTHMModelBuilder(ModelBuilder[BaseModelWrapper]):
    def __init__(self, stats: Stats, model_config: LTHMModelConfig):
        super().__init__(stats)
        self.model_config = model_config

    def build(self):
        return LTHMModelWrapper(model_config=self.model_config, stats=self.stats)