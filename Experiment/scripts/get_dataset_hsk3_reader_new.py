from argparse import ArgumentParser
import psycopg2
import os
import demjson
import numpy as np
import sys
import pandas as pd
import re
import requests
from tqdm import tqdm


def put2dicnums(rdic, key, addnum=1):
    if not rdic.get(key, ''):
        rdic[key] = addnum
    else:
        pn = rdic.get(key)
        pn += addnum
        rdic[key] = pn
    return rdic


def get_dataset_from_reader(savefile, userid, datasetype, lang):

    servertype = 'production'
    password = os.environ.get('reader_passwd')
    print('password', password)
    if servertype == 'production':
        print('load production psql')
        sybase_db = psycopg2.connect(database="django", user="reader", password='%s' % password, host="reader.cq05qytbcq26.us-west-1.rds.amazonaws.com", port="5432")
    elif servertype == 'develop':
        print('load develop psql')
        sybase_db = psycopg2.connect(database="django", user="reader", password='%s' % password, host="reader-dev.cq05qytbcq26.us-west-1.rds.amazonaws.com", port="5432")
    # 請使用簡體資料較多
    language = lang

    # ---------------取得要排除的Reader:編號 (主要是不適合當作等級評比的資料) 主要是一些 語法拼音資料 {S}--------------- #
    black_list = ['773', '774', '775', '776', '777', '778', '809', '810', '931', '934', '950', '951', '952', '953', '954', '955', '956', '957',
                  '958', '959', '960', '961', '962', '963', '964', '965', '966', '967', '968', '969', '970', '971', '972', '997', '996', '995',
                  '994', '993', '992', '991', '990', '989', '988', '987', '986', '985', '984', '983', '982', '981', '980', '979', '978', '977',
                  '976', '975', '974', '973']


    sysql_cmd = "SELECT * FROM pondlets_pondlet"
    cur = sybase_db.cursor()
    cur.execute(sysql_cmd)
    rows = cur.fetchall()
    STB2READER = {}
    READER2STB = {}

    ponddy_level_2_hsk3_level = {'0':'1', '1':'2', '2':'3', '3':'4', '4':'5', '5':'6', '6':'7-9'}
    block_readerid = []
    allow_readerid = []  # 所有STB 對應 Reader的編號清單
    for row in rows:
        # reader - pondlets_pondlet 
        # (id, simp_id, trad_id)
        # #######################
        # row (101, 3485, 3487)  (102, 1911, 1910)
        #      STB, simp, trad
        # simp -> contents_metacontent.id
        # print('row', row)
        STB2READER[row[0]] = row[2]
        READER2STB[row[2]] = row[0]
        READER2STB[row[1]] = row[0]
        if str(row[0]) in black_list:
            block_readerid.append(row[1])
            block_readerid.append(row[2])
        else:
            if language == 's':
                allow_readerid.append(row[1])
            elif language == 't':
                allow_readerid.append(row[2])
    # print('READER2STB', READER2STB)
    # ---------------取得要排除的Reader:編號 (主要是不適合當作等級評比的資料) 主要是一些 語法拼音資料 {E}--------------- #

    sybase_selectdbname = 'contents_metacontent'

    # contents_content.levelstat_hsk
    # contents_metaconten -> object_id & content_type_id (7, 9) -> contents_content

    # contents_metacontent.id 用來對應 pondlet中被block的pondlet id
    # contents_content.id => 實際線上的Readerid
    sysql_cmd = "SELECT contents_metacontent.id, contents_content.levelstat_ponddy , contents_metacontent.permanent_link, contents_metacontent.title,\
                 contents_metacontent.tags, contents_metacontent.defined_level, contents_metacontent.estimated_level, contents_metacontent.content,\
                 contents_metacontent.simp_trad, contents_metacontent.owner_id, contents_content.content_segment, contents_content.id FROM %s LEFT JOIN \
                 contents_content ON contents_content.id = contents_metacontent.object_id \
                 WHERE contents_metacontent.simp_trad IN ('t', 's') and contents_metacontent.content_type_id IN (7, 9)" \
                 % (sybase_selectdbname)

    cur = sybase_db.cursor()
    cur.execute(sysql_cmd)
    rows = cur.fetchall()
    readmax = 10000000000000
    rn = 1
    # allow_owner_ids = [1322, 828, 1001]
    # 取得的作者編號 : https://github.com/ponddy-edu/ML_pondlet_level_predictor/blob/develop/README.md
    allow_owner_ids = [int(x) for x in userid.split(';')]

    df = pd.read_csv("dictionaries_dictionary.csv")
    s_df = df.loc[:, ["s", "level"]]
    t_df = df.loc[:, ["t", "level"]]
    url = "https://api-dev.ponddy.com/api/segments/ponddy-segmenter"
    auth = f"JWT {os.environ.get('PONDDY_API_DEV_TOKEN')}"
    output = pd.DataFrame()
    wf = open('./reader_id_mapping_owenerid.txt', 'w')
    for row in tqdm(rows):
        # 最後需求欄位
        # Label,ID,Length,0,1,2,3,4,5,6,None
        # Lv.0,P101,105,15,1,0,0,0,0,0,4
        # sql field index
        # ind(0) : content id | ind(1): ponddy等級分布 | inde(2): hashid | index(3):title | index(4): tags | index(5): 自訂level (defined_level: 自訂義等級)
        # index(6): artice level(estimated_level:評估等級) | index(7): content | index(8): simp / trad | index(9): owener_id | index(10): segment
        # print('row', row)
        owener_id = row[9]
        readerid = row[0]
        content_readerid = row[11]
        simp_trad = row[8]
        content = row[7]
        current_df = s_df if simp_trad == "s" else t_df

        wf.write('readid:%s , owenerid:%s \n' % (row[0], row[9]))
        # owener_id in allow_owner_ids 創作者的清單
        # pondlets_pondlet 資料庫資料都要進入 => readid in allow_readerid
        # if owener_id in allow_owner_ids or readerid in allow_readerid:
        if owener_id in allow_owner_ids:
            # 黑名單資料不列入分析
            if readerid in block_readerid:
                continue
            print('================================================')
            print('readerid', readerid)
            print('content_readerid', content_readerid)
            ponddy_level_distributed = {"Label": "", "ID": "", "Length": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7-9": 0, "None": 0}
            print('ponddy_level_distributed', ponddy_level_distributed)

            # (defined_level: 自訂義等級)[index:5] >  (estimated_level:評估等級)[index:6]
            ponddy_defined_level = row[5]
            ponddy_estimated_level = row[6]
            ponddy_level = None
            # if ponddy_defined_level != None:
            if ponddy_defined_level is not None:
                ponddy_level = ponddy_defined_level
            else:
                ponddy_level = ponddy_estimated_level
            print('ponddy_level', ponddy_level)

            content = "".join(content.split())
            ponddy_level_distributed["Length"] = len(content)
            for index, row in df.iterrows():
                pattern = row.s if simp_trad == "s" else row.t
                level = row.level
                res = re.findall(pattern, content)
                if res:
                    ponddy_level_distributed[level] += len(res)
                    content = content.replace(pattern, "")

            r = requests.post(url=url, headers={"Authorization": auth}, json={"post": content, "pos": True, "simp_trad": simp_trad})
            if r.json():
                for sent_seg in r.json():
                    if sent_seg:
                        for word_pos in sent_seg:
                            if word_pos.endswith("PU"):
                                continue
                            ponddy_level_distributed["None"] += 1

            print('ponddy_level_distributed', ponddy_level_distributed)

            if datasetype == 'STB':
                # 標記從STB過來的資料
                if READER2STB.get(readerid, ''):
                    artice_level = ponddy_level_2_hsk3_level.get('%s' % ponddy_level)
                    ponddy_level_distributed["Label"] = f"Lv.{artice_level}"
                    ponddy_level_distributed["ID"] = 'Pondlet_%04d_%06d' % (READER2STB[readerid], readerid)
                    print("ponddy_level_distributed", ponddy_level_distributed)
                    # addline = 'Lv.%s,Pondlet_%04d_%06d,%s,%s' % (artice_level, READER2STB[readerid], readerid, len(''.join(seglists)), levelstrs)
                # 單純於Reader新增的資料
                else:
                    continue

            # 額外幫國防部(DOD)新增的資料
            elif datasetype == 'DOD':
                artice_level = ponddy_level_2_hsk3_level.get('%s' % ponddy_level)
                # addline = 'Lv.%s,dod-gloss_%06d,%s,%s' % (artice_level, readerid, len(''.join(seglists)), levelstrs)

            output = output.append(ponddy_level_distributed, ignore_index=True)

        rn += 1
        if rn > readmax:
            break
    wf.close()

    col = ["Label","ID","Length","1","2","3","4","5","6","7-9","None"]
    output = output[col]
    output = output.astype({"Length":int, "1":int, "2":int, "3":int, "4":int, "5":int, "6":int, "7-9":int, "None":int})
    output.to_csv(savefile, index=False)



if __name__ == '__main__':
    parser = ArgumentParser(prog="get_dataset_reader_new.py", description="讀取Reader中的STB / DOD()")
    parser.add_argument("--savefile", dest="savefile", help="儲存csv的路徑", default="./filename.csv")
    parser.add_argument("--userid", dest="userid", help="撈取的作者編號", default="XXX")
    parser.add_argument("--datasetype", dest="datasetype", help="資料類型", default="STB")
    parser.add_argument("--lang", dest="lang", help="語系", default="s")

    args = parser.parse_args()
    savefile = args.savefile
    userid = args.userid
    datasetype = args.datasetype
    lang = args.lang
    get_dataset_from_reader(savefile, userid, datasetype, lang)
