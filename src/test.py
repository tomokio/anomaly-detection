import os
import openpyxl
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

def _export_prediction_log(test_set):
    HEADER = ['INDEX', 'FILE_NAME', 'ANSWER', 'PREDICT', 'CORRECT']

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(HEADER)

    for sample_i, sample in enumerate(test_set):
        index     = sample_i + 1
        file_name = sample.name
        answer    = sample.y_true
        predict   = sample.y_pred
        correct   = 'o' if sample.y_true == sample.y_pred else 'x'

        ws.append([index, file_name, answer, predict, correct])

    wb.save(os.path.join(config.RESULT_DIR, 'prediction_log.xlsx'))
    wb.close()

def test(config):
    model_path  = config.MODEL_DIR + os.sep + 'svm_first_clf.pkl'
    first_clf   = load_model(model_path)

    model_path  = config.MODEL_DIR + os.sep + 'svm_second_clf.pkl'
    second_clf  = load_model(model_path)

    test_set    = load_test_set(config)
    test_set_X  = [sample.X for sample in test_set]

    pred_list   = first_clf.predict(test_set_X)
    pred_list_2 = second_clf.predict(test_set_X)

    if config.ENABLE_SECOND_CLF:
        for pred_i, prediction in enumerate(pred_list):
            if prediction == -1:
                pred_list[pred_i] = pred_list_2[pred_i]

    for sample_i, sample in enumerate(test_set):
        sample.y_pred = pred_list[sample_i]

    _calc_score(test_set)
    _export_prediction_log(test_set)


if __name__ == '__main__':

    config  = load_config()
    test(config)
