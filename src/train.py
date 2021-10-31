import os
from   utils                 import save_model
from   config                import load_config
from   dataset               import load_train_set
from   sklearn.svm           import OneClassSVM
from   sklearn.pipeline      import make_pipeline
from   sklearn.preprocessing import StandardScaler


def train(config):
    train_set   = load_train_set(config)
    train_set_X = [sample.X for sample in train_set]

    clf = make_pipeline(StandardScaler(), OneClassSVM(kernel=config.KERNEL, gamma=config.GAMMA, nu=config.NU))
    clf.fit(train_set_X)

    model_path = config.MODEL_DIR + os.sep + 'svm_trained_model.pkl'
    save_model(clf, model_path)


if __name__ == '__main__':

    config  = load_config()
    train(config)
