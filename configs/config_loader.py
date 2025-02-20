from typing import Optional, Dict, Any
import yaml
import os

def load_config(config_path:str="config.yaml") -> Optional[Dict[str, Any]]:
    """Loads configuration from a YAML file"""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found")
        return None
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

def get_params(config_path: str="config.yaml") -> Optional[Dict[str, Any]]:
    """Loads and returns the config parameters. Handles missing sections"""
    config = load_config()
    if config is None:
        return None
    params_dict = {}
    params_dict["paths"] = config.get("paths")
    params_dict["training"] = config.get("training")
    params_dict["logging"] = config.get("logging")
    params_dict["tensorrt"] = config.get("tensorrt")
    if not params_dict.get("paths"):
        print("Warning: 'paths' section not found in config.")
    if not params_dict.get("training"):
        print("Warning: 'training' section not found in config.")
    if not params_dict.get("logging"):
        print("Warning: 'logging' section not found in config.")
    if not params_dict.get("tensorrt"):
        print("Warning: 'tensorrt' section not found in config.")
    else:
        print("Warning: 'tensorrt' section not found in config.")
    return params_dict