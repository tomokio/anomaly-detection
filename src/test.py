import os
from   utils           import load_model
from   config          import load_config
from   dataset         import load_test_set
from   contextlib      import redirect_stdout
from   sklearn.metrics import classification_report


def _calc_score(test_set):
    y_true = [sample.y_true for sample in test_set]
    y_pred = [sample.y_pred for sample in test_set]
    print(classification_report(y_true, y_pred, target_names=['-1', '1']))

    os.makedirs(config.RESULT_DIR, exist_ok=True)
    log_file_path = config.RESULT_DIR + os.sep + 'result.txt'

    with open(log_file_path, 'w') as log_file:
        with redirect_stdout(log_file):
            print(classification_report(y_true, y_pred, target_names=['-1', '1']))

def test(config):
    model_path    = config.MODEL_DIR + os.sep + 'svm_trained_model.pkl'
    trained_model = load_model(model_path)

    test_set      = load_test_set(config)
    test_set_X    = [sample.X for sample in test_set]

    pred_list     = trained_model.predict(test_set_X)
    for sample_i, sample in enumerate(test_set):
        sample.y_pred = pred_list[sample_i]

    _calc_score(test_set)


if __name__ == '__main__':

    config  = load_config()
    test(config)
