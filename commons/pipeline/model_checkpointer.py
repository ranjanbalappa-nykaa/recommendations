
from typing import Callable, Optional, Dict, Any

import pandas as pd


class ModelCheckpointer:
    def __init__(
            self,
            model_exporter: Callable[[Any, pd.DataFrame, pd.DataFrame], None]
    ):
        self.model_exporter = model_exporter

    def checkpoint(self, state_dict: Optional[Dict[str, Any]] = None, result_df: Optional[pd.DataFrame] = None):
        self.model_exporter(state_dict, result_df)
