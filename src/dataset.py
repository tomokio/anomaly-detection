import utils
import random
import librosa
import numpy as np
from   dataclasses import dataclass


@dataclass
class Sample:
    name:   str
    X:      np.ndarray
    y_true: np.float64
    y_pred: np.float64

def _extract_feature(config, sample_path):
    sample, sr  = librosa.load(path=sample_path, offset=1.5, duration=4, sr=None, mono=True)

    if   config.EXTRACTION_METHOD == 'power':
        feature = librosa.amplitude_to_db(np.abs(librosa.stft(sample, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, center=False)), ref=np.max)

    elif config.EXTRACTION_METHOD == 'mel':
        feature = librosa.power_to_db(librosa.feature.melspectrogram(sample, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, \
                                                                     center=False, power=2.0, n_mels=config.N_MELS), ref=np.max)

    else:
        raise ValueError('Unsupported EXTRACTION_METHOD selected.')

    return feature.reshape(-1)

def load_train_set(config):
    train_set   = list()
    sample_list = utils.glob_dir(config.DATASET_DIR, 'train*/*.wav')

    random.seed(0)
    random.shuffle(sample_list)

    for sample in sample_list:
        X = _extract_feature(config, sample)
        train_set.append(Sample(name=sample.name, X=X, y_true=1, y_pred=None))

    return train_set

def load_test_set(config):
    test_set     = list()
    test_normal  = utils.glob_dir(config.DATASET_DIR, 'test_normal/*.wav')
    test_anomaly = utils.glob_dir(config.DATASET_DIR, 'test_anomaly/*.wav')

    random.seed(0)
    random.shuffle(test_normal)
    random.shuffle(test_anomaly)

    if   config.TESTING_TYPE == 'dev':
        sample_list = test_normal[:config.DEV_NORMAL_SAMPLES] + test_anomaly[:config.DEV_ANOMALY_SAMPLES]
    elif config.TESTING_TYPE == 'final':
        sample_list = test_normal[config.DEV_NORMAL_SAMPLES:] + test_anomaly[config.DEV_ANOMALY_SAMPLES:]
    else:
        raise ValueError('Unsupported TESTING_TYPE selected.')

    random.shuffle(sample_list)

    for sample in sample_list:
        X = _extract_feature(config, sample)

        if   'ab' in str(sample):
            y_true = -1
        elif 'normal' in str(sample):
            y_true = 1
        else:
            raise ValueError('Failed to assign answer label.')

        test_set.append(Sample(name=sample.name, X=X, y_true=y_true, y_pred=None))

    return test_set
