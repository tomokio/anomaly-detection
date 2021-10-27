import pickle
import pathlib


def glob_dir(path_: str, query_: str) -> list:
    return list(pathlib.Path(path_).glob(query_))

def save_model(model_obj, save_path: str):
    pickle.dump(model_obj, open(save_path, 'wb'))

def load_model(load_path: str):
    model_obj = pickle.load(open(load_path, 'rb'))
    return model_obj
