import yaml
import joblib

CONFIG_DIR = "../config/config.yaml"

# load config
def load_config() -> dict: 
    try:
        with open(CONFIG_DIR, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as fe:
        raise RuntimeError("Parameters file not found in path.")

    return config

# pickle load
def pickle_load(file_path: str):
    return joblib.load(file_path)

# pickle dump
def pickle_dump(data, file_path: str) -> None:
    joblib.dump(data, file_path)

print('utils good')