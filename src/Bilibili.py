"""
First created in 10/2021
Updated on 26/09/2022
Author: Chizuru Chionghwa
"""
import json
import time

import pymysql
import requests

# MYSQL数据库连接 (需要密码和数据库名称)
conn = pymysql.connect(host='127.0.0.1', user='root', password='', database='bilibili')
cur = conn.cursor()

# 建立Session
s = requests.Session()

# 登录Bilibili后使用的SESSDATA
cookies = {"SESSDATA": "7ae91bf0%2C1667630231%2Caabb2*51"}

# requests使用的headers
headers = {
    'User-agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36 Edg/90.0.818.62',
}


# 获取动态列表
def get_dynamic_list(mid, start_offset):
    # 初始化next_offset 从第1页开始
    next_offset, pageNum = start_offset, 1

    # type 转发动态17 相簿(图片动态)11 文字动态17 视频1 专栏12
    # rid 动态id 图片动态:相簿id 文字动态:动态id 视频:avid 专栏:cvid
    # type详细具体见: https://github.com/SocialSisterYi/bilibili-API-collect/blob/master/comment/readme.md#%E8%AF%84%E8%AE%BA%E5%8C%BA%E7%B1%BB%E5%9E%8B%E4%BB%A3%E7%A0%81

    # 循环收集动态列表
    while 1:
        # 动态列表的API格式
        link = f"https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/space?offset={next_offset}&host_mid={mid}&timezone_offset=-480"
        r = s.get(link, headers=headers)
        info = json.loads(r.text)
        print(link)
        next_offset = info["data"]["offset"]

        # 输出当前页码
        print(pageNum)

        # 收集每条动态的相关信息
        for dt in info["data"]["items"]:

            oid = dt["id_str"]  # str
            type = dt["basic"]["comment_type"]  # int
            rid = dt["basic"]["rid_str"]  # str

            # 以后更新新的动态的时候可以使用这个来更新，一到已经收录的动态则return
            try:
                cur.execute(f"""INSERT INTO DYNAMIC_LIST(MID,OID,RID,TYPE)VALUES({mid},'{oid}','{rid}',{type})""")
            except:
                return None

        # 继续下一页
        pageNum += 1

        # 每页休息1s
        time.sleep(1.0)


def get_comment_list(oid, type, mid):
    pageNum = 1
    while 1:
        link = f"http://api.bilibili.com/x/v2/reply?type={type}&oid={oid}&ps=49&pn={pageNum}"
        r = s.get(link, headers=headers)
        info = json.loads(r.text)
        print(link)
        print(pageNum)

        if info['code'] != 0:
            if info["code"] == -412:
                print("主拦截")
                print(link)
                time.sleep(1200)
                continue
            if info['code'] == -404:
                print('链接有误')
                break
            if info['code'] == 12002:
                print("评论区已关闭")
                break
        if info["data"]["replies"] == []:
            break
        for comment in info["data"]["replies"]:
            rpid = comment['rpid']
            user_mid = comment['mid']
            root = comment['root']
            parent = comment['parent']
            timestamp = comment["ctime"]
            msg = comment["content"]["message"]

            # 主评论的加入,如果已加入则跳过
            try:
                cur.execute(
                    f"""INSERT INTO COMMENT_LIST(RPID,RID,USER_MID,UP_MID,ROOT,PARENT,TIME,MSG)VALUES('{rpid}','{oid}',{user_mid},{mid},'{root}','{parent}',{timestamp},'{msg}')""")
            except:
                continue

        pageNum += 1
        time.sleep(0.5)


def create_table():
    # 动态列表的table
    # OID:动态id TYPEA TIME:创建时间 MID:UP主id TYPEB RID:视频专栏id或动态id
    # cur.execute("""
    # CREATE TABLE DYNAMIC_LIST (
    # MID INT NOT NULL,
    # OID CHAR(100),
    # RID CHAR(100),
    # TYPE INT NOT NULL,
    # PRIMARY KEY(OID))
    # ENGINE=MyISAM
    # """)

    # 主评论列表的table
    # RPID:主评论id ...
    cur.execute("""
    CREATE TABLE COMMENT_LIST (
    RPID  CHAR(100) NOT NULL,
    RID  CHAR(100),
    USER_MID INT,
    UP_MID INT,
    ROOT CHAR(100),
    PARENT CHAR(100),
    TIME INT,
    MSG VARCHAR(1000),
    PRIMARY KEY(RPID,RID))
    ENGINE=MyISAM
    """)


def dynamic_sample():
    # 四禧丸子_Official 1129115529
    # 又一充电中 1217754423
    # 恬豆发芽了 1660392980
    # 沐霂是MUMU呀 1878154667
    # 梨安不迷路 1900141897

    # 教程:
    # Step1: 访问 https://space.bilibili.com/{mid}/dynamic
    # Step2: F12 Network界面 搜索offset
    # Step3: 下拉动态 直至出现第一个offset
    # create_table()
    members_list = [
        (1129115529, 706438127970943008),
        (1217754423, 706799759923347479),
        (1660392980, 707612698462912514),
        (1878154667, 707462752541605895),
        (1900141897, 707614201690980471)]
    for member, start_offset in members_list:
        get_dynamic_list(member, start_offset)


def comment_sample():
    # create_table()
    cur.execute("""SELECT * FROM DYNAMIC_LIST""")
    results = cur.fetchall()
    flag = 0
    for dynamic in results:
        if dynamic[2] == "724054613":
            flag = 1
        if flag == 1:
            if dynamic[3] == 17:
                get_comment_list(dynamic[1], dynamic[3], dynamic[0])
            else:
                get_comment_list(dynamic[2], dynamic[3], dynamic[0])


def comment_sample2():
    create_table()
    avs = [555434722, 940746045, 257676702, 385486211, 898196558, 555548194, 301800940]
    for av in avs:
        get_comment_list(av, 1, 0)


def main():
    comment_sample2()
    # dynamic_sample()


if __name__ == "__main__":
    main()
