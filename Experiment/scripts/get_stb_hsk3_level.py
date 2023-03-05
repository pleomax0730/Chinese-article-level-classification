from argparse import ArgumentParser
import os
from stbdb import db
import numpy as np


def generate_level_from_db(savefile, stb_title_field, stb_level_field):

    sqlcmd = "SELECT * FROM dictionaries_dictionary"
    records = db.query(sqlcmd).fetchall()

    pondlet_level_to_diclevels = {}
    totalevels = []
    for rec in records:
        word = rec.get(stb_title_field)
        level = rec.get(stb_level_field)
        levelv = "%s" % level
        if levelv:
            pondlet_level_to_diclevels[word] = "%s" % levelv
            if levelv not in totalevels:
                totalevels.append(levelv)
    print("pondlet_level_to_diclevels", pondlet_level_to_diclevels)
    print("len records", len(records))
    np.save(savefile, pondlet_level_to_diclevels)
    print("%s is generate!" % savefile)
    print("totalevels", totalevels)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="get_dataset_reader.py", description="讀取Reader中的STB / DOD()"
    )
    parser.add_argument(
        "--savefile", dest="savefile", help="儲存的特徵路徑", default="./save_feaure.txt"
    )
    parser.add_argument(
        "--stb_title_field", dest="stb_title_field", help="撈取的標題欄位", default="simp"
    )
    parser.add_argument(
        "--stb_level_field",
        dest="stb_level_field",
        help="撈取的等級欄位",
        default="level_hsk3",
    )

    args = parser.parse_args()
    savefile = args.savefile
    stb_title_field = args.stb_title_field
    stb_level_field = args.stb_level_field
    generate_level_from_db(savefile, stb_title_field, stb_level_field)
