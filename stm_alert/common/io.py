from pathlib import Path
from typing import Union
import yaml


def load_yaml_from_path(path: Union[str, Path]) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)
