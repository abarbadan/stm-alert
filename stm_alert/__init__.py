from pathlib import Path


def get_path_from_project_root() -> Path:
    return Path(__file__).parent
