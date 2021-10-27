import yaml


class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def load_config():
    with open('config.yml', 'r') as config_file:
        config_dict = yaml.safe_load(config_file)

    config = Config(config_dict)

    return config
