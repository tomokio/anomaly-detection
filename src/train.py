import os
import copy
from   utils                 import save_model
from   config                import load_config
from   dataset               import load_train_set
from   sklearn.svm           import OneClassSVM
from   sklearn.pipeline      import make_pipeline
from   sklearn.preprocessing import StandardScaler


def train(config):
    train_set   = load_train_set(config)
    train_set_X = [sample.X for sample in train_set]

    first_clf   = make_pipeline(StandardScaler(), OneClassSVM(kernel=config.KERNEL, gamma=config.GAMMA, nu=config.NU))
    second_clf  = copy.deepcopy(first_clf)

    first_clf.fit(train_set_X)

    retrain_list = list()
    for sample_i, prediction in enumerate(first_clf.predict(train_set_X)):
        if prediction == -1:
            retrain_list.append(train_set_X[sample_i])

    second_clf.fit(retrain_list)

    model_path = config.MODEL_DIR + os.sep + 'svm_first_clf.pkl'
    save_model(first_clf, model_path)

    model_path = config.MODEL_DIR + os.sep + 'svm_second_clf.pkl'
    save_model(second_clf, model_path)


if __name__ == '__main__':

    config  = load_config()
    train(config)
