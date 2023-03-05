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
source .env

# 取的STB的資料 : Owner_id in (1322,828)，讀取簡繁的資料
# ponddy_level : 2021.06.23 (simp/trad) : 1617
python scripts/get_dataset_reader.py --savefile ./datasets/pondlet_STB_2021-all.csv --userid "1322;828" --datasetype STB

# 取得Gloss的資料 DOD(Gloss Owner_id = 1001)，讀取簡體的資料(簡體資料較多)
# ponddy_level : 2021.06.23 (simp/trad) : 120
python scripts/get_dataset_reader.py --savefile ./datasets/pondlet_gloss_2021-all.csv --userid "1001" --datasetype DOD

# 合併兩份 csv 至 pondlet_gloss-2022-all.csv
cd datasets
python combine_pondlet_gloss.py \
--input_csv pondlet_STB_2021-all.csv pondlet_gloss_2021-all.csv \
--output_csv pondlet_gloss-2022-all.csv

# 會額外輸出 pondlet_gloss-2022-all.csv 的 train/test csv 檔
# datasets/pondlet_STB_pondlet_20220803_content_data_train.csv
# datasets/pondlet_STB_pondlet_20220803_content_data_test.csv
python bert_preprocessing.py \
--input_csv pondlet_gloss-2022-all.csv \
--output_csv pondlet_STB_pondlet_20220803_content_data.csv
```

## 訓練

cd scripts
python bert_training_pondlet.py
