# BERT training workflow

## 建立環境

```bash=
python3.8 -m venv venv38
source venv38/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## 取得訓練資料

```bash=

# 啟動環境變數
source .env.hsk3

python3 scripts/get_dataset_hsk3_reader_new.py \
--savefile ./datasets/pondlet_STB_HSK3_20220429_new2.csv \
--userid "1322;828" \
--datasetype STB

# 修正老師提供的 Label
cd datasets
python3 label_correction.py \
--input_csv pondlet_STB_HSK3_20220429_new.csv \
--output_csv pondlet_STB_HSK3_20220614_new_with_review_label.csv

# 會額外輸出 pondlet_STB_HSK3_20220714_content_data 的 train/test csv 檔
# datasets/pondlet_STB_HSK3_20220714_content_data_train.csv
# datasets/pondlet_STB_HSK3_20220714_content_data_test.csv
python3 bert_preprocessing.py \
--input_csv pondlet_STB_HSK3_20220614_new_with_review_label.csv \
--output_csv pondlet_STB_HSK3_20220714_content_data.csv

```

## 訓練

cd scripts
python bert_training_hsk3.py
