from stm_alert import get_path_from_project_root
import yaml


def load_config(config_name):
    config_path = get_path_from_project_root() / f"gtfs/configs/{config_name}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
