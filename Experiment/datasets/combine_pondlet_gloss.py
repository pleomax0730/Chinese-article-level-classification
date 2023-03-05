import pandas as pd
from argparse import ArgumentParser


def main(args):
    print(args.input_csv)
    df = pd.concat(map(pd.read_csv, args.input_csv), ignore_index=True)
    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="get_dataset_reader_new.py", description="讀取Reader中的STB / DOD()"
    )
    parser.add_argument(
        "--input_csv",
        help="pondlet以及gloss的兩份csv",
        nargs="+",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_csv",
        help="pondlet以及gloss的合併csv",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    main(args)
