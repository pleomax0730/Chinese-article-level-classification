from utils import process_dataset
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pycaret.classification import *
import numpy as np


def do_train(datasets):
    balance_data = process_dataset(datasets)
    balance_data.drop("ID", inplace=True, axis=1)
    exp_mclf101 = setup(data = balance_data, target = 'Label', session_id=42)
    best = compare_models()
    # X = balance_data.iloc[:, 1:].to_numpy()
    # le = preprocessing.LabelEncoder()
    # le.fit(balance_data.Label)
    # balance_data['categorical_label'] = le.transform(balance_data.Label)
    # y = balance_data.iloc[:, -1].to_numpy()

    # print(le.classes_) # ['Lv.1' 'Lv.2' 'Lv.3' 'Lv.4' 'Lv.5' 'Lv.6']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
    '''
    balance_data.shape # (1735, 10)
    balance_data["Label"].value_counts()
    Lv.4    624
    Lv.5    331
    Lv.3    282
    Lv.6    214
    Lv.2    148
    Lv.1    136
    '''
    


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

    do_train(datasets)


if __name__ == '__main__':
    main()