import json
import torch
from pathlib import Path

def print_gpu_is_used() -> None:
    """ Print banner to show if gpu is used. """
    # Check if GPU available
    if torch.cuda.is_available():
        print("\n###################################################")
        print("Using GPU for training.")
        print("###################################################\n")
    else:
        print("GPU not available, using CPU instead.")


def get_config_env(config_path: str) -> dict[str, str]:
    """ Load config env """
    config_path = Path(config_path)
    if not config_path.exists() or not config_path.is_file():
        raise NameError(f"Config file not found for path {config_path}")

    with open(config_path, 'r') as file:
        config_env: dict[str, str] = json.load(file)
    
    return config_env