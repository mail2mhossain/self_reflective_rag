import yaml
import os

# Define the path to the config.yaml file
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yaml'))

def get_config():
    # Open and read the YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        return config

