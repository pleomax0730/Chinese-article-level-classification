import pandas as pd
import re
import requests
import os
from tqdm import tqdm

url = "https://api-dev.ponddy.com/api/segments/ponddy-segmenter"
auth = f"JWT {os.environ.get('PONDDY_API_DEV_TOKEN')}"
df = pd.read_csv("dictionaries_dictionary.csv")
df = df.loc[:, ["s", "level"]]
hsk3_level_distribution = {"Label": "", "ID": "", "Length": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7-9": 0, "None": 0}

test_content = """每当我回忆起那次面试经历时，总感觉是刚发生的事，让我还是记忆犹新，心跳加速。
当我坐在门外不停跺着脚焦急地等待时，心里不停地默念：“你行的，前面都已经过关斩将，最后这关一定没有问题的，常见问题都练习了十几遍。”当我念到我的名字时，我被惊了一下。可当我面带微笑推开门的那一刻，我定住了，被眼前齐刷刷看我的面试官吓到了，“这么多啊！完了！”我的笑容一下子僵住了。本以为先是自我介绍，却是开门见山地来了一个专业英语问题，我懵住了，脑子一下短路，不知道问了啥，我只能尴尬地叫面试者重新说一遍，这时我的双手死死地握着，心跳停止，屏住呼吸，回答时不知所云，完全乱猜。下一个问题还是不放过我，而且是死死地问我好几个问题，一环扣一环，我只感觉没有喘气的余地。“完了，这下真的完了！”心里不停有个小声音提醒自己，连自己都不知道回答的什么。最后，问的什么问题完全不知道，干脆破罐子破摔，随便回答了，完全没有经过大脑思考，我不敢看面试官的表情，因为我已经知道了结局，那时我只想赶紧出去，赶紧离开这个是非之地。最后我怎么出来的，我已不知道了，只知出门后感觉自己又能呼吸了，一直颤抖的心一下子又平静了。
后来也慢慢明白了：或许人生就是由这些失败的石头，一点一点铺成了最后通往成功那条幸福的小路，让自己更加坚强，也因为接受每一次的失败经验，让我们知道自己的不足，同时也接受自己 ，让自己成长，让自己的人生更加完整。"""
test_content = "".join(test_content.split())
hsk3_level_distribution["Length"] = len(test_content)
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    pattern = row.s
    level = row.level
    res = re.findall(pattern, test_content)
    if res:
        print("res", res)
        hsk3_level_distribution[level] += len(res)
        test_content = test_content.replace(pattern, "")

r = requests.post(url=url, headers={"Authorization": auth}, json={"post": test_content, "pos": True, "simp_trad": "s"})
if r.json():
    for sent_seg in r.json():
        if sent_seg:
            for word_pos in sent_seg:
                if word_pos.endswith("PU"):
                    continue
                hsk3_level_distribution["None"] += 1

print(test_content)
print(hsk3_level_distribution)
output = pd.DataFrame()
output = output.append(hsk3_level_distribution, ignore_index=True)
# output = output.astype({"Length":int, "1":int, "2":int, "3":int, "4":int, "5":int, "6":int, "7-9":int, "None":int})
