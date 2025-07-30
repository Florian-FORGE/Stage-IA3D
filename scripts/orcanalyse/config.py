import os
import yaml

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to Config.yaml
config_path = os.path.join(script_dir, "Config.yaml")

# Load the YAML file
with open(config_path, "r") as file:
    config_data = yaml.safe_load(file)