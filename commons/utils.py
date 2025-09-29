import importlib
import os
import requests




def instantiate_class(full_path: str, *args, **kwargs):
    module_name, clazz_name = full_path.rsplit('.', maxsplit=1)
    module = importlib.import_module(module_name)
    clazz = getattr(module, clazz_name)
    return clazz(*args, **kwargs)