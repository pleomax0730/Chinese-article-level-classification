import os
from argparse import ArgumentParser

import pandas as pd
from pony.orm import *
from sklearn.model_selection import train_test_split


def read_content_from_db(pondlet_ids):
    db = Database()
    db.bind(
        provider="postgres",
        user="reader",
        password=f"{os.environ.get('reader_passwd', '')}",
        host="reader.cq05qytbcq26.us-west-1.rds.amazonaws.com",
        database="django",
        port=5432,
    )
    content_list = []
    with db_session:
        # data = db.select(
        #     "select id, content from contents_metacontent where id = 3 or id = 4"
        # )

        metacontent_ids = [int(p.split("_")[-1]) for p in pondlet_ids]
        for metacontent_id in metacontent_ids[:]:
            data = db.select(
                "select content from contents_metacontent where id = $metacontent_id"
            )
            content_list.append(data[0].replace("\n", "").strip())
    return content_list


def main(args):
    df = pd.read_csv(args.input_csv)
    pondlet_ids = df.loc[:, "ID"].to_list()
    content_list = read_content_from_db(pondlet_ids)
    df["content"] = content_list

    new_df = df.loc[:, ["Label", "Length", "content"]]
    # pondlet_STB_HSK3_20220714_content_data.csv
    new_df.to_csv(args.output_csv, index=False)

    new_df.drop(new_df[new_df.Label == "Lv.7-9"].index, inplace=True, axis=0)

    df_train, df_test = train_test_split(
        new_df, test_size=0.1, stratify=new_df.Label, random_state=42
    )
    df_train = df_train.rename(columns={"Label": "labels"})
    df_test = df_test.rename(columns={"Label": "labels"})

    train_filename = args.output_csv.replace(".csv", "_train.csv")
    test_filename = args.output_csv.replace(".csv", "_test.csv")
    df_train.to_csv(train_filename, index=False)
    df_test.to_csv(test_filename, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    main(parser.parse_args())
