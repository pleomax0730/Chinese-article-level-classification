from argparse import ArgumentParser
import psycopg2
import os
import demjson
import numpy as np


def put2dicnums(rdic, key, addnum=1):
    if not rdic.get(key, ''):
        rdic[key] = addnum
    else:
        pn = rdic.get(key)
        pn += addnum
        rdic[key] = pn
    return rdic


def get_dataset_from_reader(savefile, userid, datasetype, lang, hsk3_simp, hsk3_trad):

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

    hsk3_simp_dictionary = {}
    hsk3_simp_dictionary = np.load(hsk3_simp, allow_pickle=True).item()
    hsk3_trad_dictionary = {}
    hsk3_trad_dictionary = np.load(hsk3_trad, allow_pickle=True).item()

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

    linelists = ['Label,ID,Length,1,2,3,4,5,6,7-9,None']
    wf = open('./reader_id_mapping_owenerid.txt', 'w')
    for row in rows:
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
        simp_trad = row[9]
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
            ponddy_level_distributed = {}
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

            hashid = row[2]
            print('hashid', hashid)
            owener_id = row[9]
            print('owener_id', owener_id)
            # print('segments', row[10])
            segments_string = row[10]
            # print('segments_string:', segments_string )
            # print('type segments_string', type(segments_string))
            # 髒髒包讀出來是 list 格式
            if type(segments_string) == list:
                # {'id': 2258, 'pos': 'n.', 'show': False, 'word': '房子', 'levels': {'hsk': 'None', 'tocfl': '0', 'ponddy': '0'}, 'pinyin': 'fángzi'}
                seglists = []
                # print('segments_string', segments_string)
                for lv1dic in segments_string:
                    # print('type(dic)', type(dic))
                    # print('dic', dic)
                    # print('lv1dic', lv1dic)
                    for dic in lv1dic.get('segments'):
                        thisword = dic.get('word')
                        if simp_trad == 'simp':
                            ponddy_level_distributed = put2dicnums(ponddy_level_distributed, hsk3_simp_dictionary.get(thisword, 'None'), addnum=1)
                        else:
                            ponddy_level_distributed = put2dicnums(ponddy_level_distributed, hsk3_trad_dictionary.get(thisword, 'None'), addnum=1)
                        seglists.append(thisword)
                # print('seglists--1', seglists)
            elif segments_string is None:
                continue
            # 髒髒包讀出來是 str 格式
            else:
                segments = demjson.decode(str(segments_string))
                # print('segments', segments)
                # segments = json.loads()
                seglists = []
                for lv1dic in segments:
                    # print('lst', lst)
                    for dic in lv1dic.get('segments'):
                        thisword = dic.get('word')
                        if simp_trad == 'simp':
                            ponddy_level_distributed = put2dicnums(ponddy_level_distributed, hsk3_simp_dictionary.get(thisword, 'None'), addnum=1)
                        else:
                            ponddy_level_distributed = put2dicnums(ponddy_level_distributed, hsk3_trad_dictionary.get(thisword, 'None'), addnum=1)                        
                        seglists.append(thisword)
                # print('seglists--2', seglists)
            # Lv.0,P101,105,15,1,0,0,0,0,0,4
            # ponddy_level_distributed {'0': 60, '1': 23, '2': 30, '3': 15, '4': 10, '5': 5, '6': 0, 'None': 16}
            levelsts = []
            print('ponddy_level_distributed', ponddy_level_distributed)
            # ['None', '4', '7-9', '2', '6', '1', '3', '5']
            for xr in ['1', '2', '3', '4', '5', '6', '7-9', 'None']:
                levelsts.append('%s' % ponddy_level_distributed.get(xr, '0'))
            levelstrs = ','.join(levelsts)
            if datasetype == 'STB':
                # 標記從STB過來的資料
                if READER2STB.get(readerid, ''):
                    artice_level = ponddy_level_2_hsk3_level.get('%s' % ponddy_level)
                    addline = 'Lv.%s,Pondlet_%04d_%06d,%s,%s' % (artice_level, READER2STB[readerid], readerid, len(''.join(seglists)), levelstrs)
                # 單純於Reader新增的資料
                else:
                    continue
                    # addline = 'Lv.%s,Readeronly_%06d,%s,%s' % (ponddy_level, readerid, len(''.join(seglists)), levelstrs)
            # 額外幫國防部(DOD)新增的資料
            elif datasetype == 'DOD':
                artice_level = ponddy_level_2_hsk3_level.get('%s' % ponddy_level)
                addline = 'Lv.%s,dod-gloss_%06d,%s,%s' % (artice_level, readerid, len(''.join(seglists)), levelstrs)
            print('addline', addline)
            linelists.append(addline)
        rn += 1
        if rn > readmax:
            break
    wf.close()

    with open(savefile, "w") as fh:
        for line in linelists:
            fh.write("%s\n" % line)
    print('%s is generate! total:%s' % (savefile, len(linelists)))


if __name__ == '__main__':
    parser = ArgumentParser(prog="get_dataset_reader.py", description="讀取Reader中的STB / DOD()")
    parser.add_argument("--savefile", dest="savefile", help="儲存的特徵路徑", default="./save_feaure.txt")
    parser.add_argument("--userid", dest="userid", help="撈取的作者編號", default="XXX")
    parser.add_argument("--datasetype", dest="datasetype", help="資料類型", default="STB")
    parser.add_argument("--lang", dest="lang", help="語系", default="s")
    parser.add_argument("--hsk3_simp", dest="hsk3_simp", help="Hsk3 level(simp) numpy cache file", default="")
    parser.add_argument("--hsk3_trad", dest="hsk3_trad", help="Hsk3 level(trad) numpy cache file", default="")
    
    args = parser.parse_args()
    savefile = args.savefile
    userid = args.userid
    datasetype = args.datasetype
    lang = args.lang
    hsk3_simp = args.hsk3_simp
    hsk3_trad = args.hsk3_trad
    get_dataset_from_reader(savefile, userid, datasetype, lang, hsk3_simp, hsk3_trad)
