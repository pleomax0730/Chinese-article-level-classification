import os
import os.path
import joblib
import mlflow
import glob
from argparse import ArgumentParser
from models import train_model
from utils import process_dataset
import lightgbm
import lightgbm as lgb
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()

MLFLOW_S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')
# http://XXXXXX:8002
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI')
print('====MLFLOW_TRACKING_URI====:%s' % MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MLFLOW_EXPERIMENTS_NAME = os.environ.get('MLFLOW_EXPERIMENTS_NAME', '')
mlflow.set_experiment(MLFLOW_EXPERIMENTS_NAME)
# s3://mlflow-models/XX/b7d55ee029c3454eae375f568a5fd419/artifacts
artifact_path_s3 = mlflow.get_artifact_uri()
print('artifact_path_s3:', artifact_path_s3)
print(mlflow.get_artifact_uri())


def do_train(model_output, model_prefix, model_type, datasets):
    print('model_type', model_type)
    balance_data = process_dataset(datasets)

    '''
    balance_data.head()

        Label                ID  Length    1   2   3   4  5   6  7-9  None
    0  Lv.3  dod-gloss_002763     418   65  18   4  16  1  11    9   148
    1  Lv.4  dod-gloss_002748     255   20   2   8   5  3   5    4    92
    2  Lv.5  dod-gloss_002733     440   35  16  15   8  3   0    4   178
    3  Lv.6  dod-gloss_002725     719  119  20  12  17  5   3   16   290
    4  Lv.6  dod-gloss_002716    1137  135  29  25  22  3  12   17   437
    '''    

    confustion_martix_files = glob.glob('./reports/confustion*')
    models_files = glob.glob('./reports/*.h5')
    for f in confustion_martix_files + models_files:
        try:
            os.remove(f)
        except Exception as e:
            _ = e  # noqa: F841

    mlflow.set_tag("mlflow.user", os.environ.get('MLFLOW_MYSET_USER', 'owen'))
    mlflow.log_param(".r__model_type", model_type)
    dt = train_model(balance_data, model_type)
    print('accuracy(train)', dt['accuracy(train)'])
    print('accuracy(test)', dt['accuracy(test)'])
    mlflow.log_metric("train_acc", round(dt['accuracy(train)'], 2))
    mlflow.log_metric("valid_acc", round(dt['accuracy(test)'], 2))
    mlflow.log_param(".r__train_acc", round(dt['accuracy(train)'], 2))
    mlflow.log_param(".r__valid_acc", round(dt['accuracy(test)'], 2))    
    # 儲存模型
    save_model_path = '%s/%s.h5' % (model_output, model_prefix)
    save_model_s3_path = 'reports/%s.h5' % (model_prefix)

    joblib.dump(dt['kernel'], save_model_path)
    joblib.dump(dt['kernel'], save_model_s3_path)
    print('model : %s is generate' % save_model_path)
    # 將reports 路徑上傳到主機上
    # mlflow.log_artifacts(save_model_path, "sklearn_sklearn.h5")

    mlflow.log_artifacts("reports")


def main():
    parser = ArgumentParser(prog="train.py", description="train for pondlet level predict")
    parser.add_argument("--model_output", dest="model_output", help="模型輸出的路徑", default="./Models")
    parser.add_argument("--model_type", dest="model_type", help="模型類型", default="DecisionTreeClassifier")
    parser.add_argument("--model_prefix", dest="model_prefix", help="模型儲存開頭命名", default="")
    parser.add_argument("--datasets", dest="datasets", help="讀取的訓練資料清單，多個檔案路徑以;區隔", default="")
    args = parser.parse_args()
    model_output = args.model_output
    model_prefix = args.model_prefix
    model_type = args.model_type
    datasets = args.datasets

    do_train(model_output, model_prefix, model_type, datasets)


if __name__ == '__main__':
    main()
