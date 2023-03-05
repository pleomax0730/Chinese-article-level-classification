from argparse import ArgumentParser

import pandas as pd


def main(args):
    # 老師修正過後的 Label
    review_label_df = pd.read_csv(
        "HSK3 Content Level Review & Override (20220516) - Missed.csv"
    )

    # 要修正的資料
    df = pd.read_csv(args.input_csv)

    for _, review in review_label_df.iterrows():
        ID = review["ID"]
        idx = df.loc[df["ID"] == ID].index
        df.iloc[idx, 0] = review["Reviewed Label"]
        # print(df.loc[df["ID"] == ID].Label)

    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    main(parser.parse_args())
