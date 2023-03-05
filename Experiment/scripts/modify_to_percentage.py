# %%
import pandas as pd

df = pd.read_csv("/home/vincent0730/ML_pondlet_level_predictor/datasets/pondlet_STB_HSK3_20220614_new_with_review_label.csv")
df.head()
# %%
row_sum = df.iloc[:, 3:].sum(axis=1)
print("row_sum", row_sum)

for i in range(3, 11):
    df.iloc[:, i] = df.iloc[:, i] / row_sum

df.to_csv("/home/vincent0730/ML_pondlet_level_predictor/datasets/pondlet_STB_HSK3_20220708_percentage.csv", index=False)
# %%
