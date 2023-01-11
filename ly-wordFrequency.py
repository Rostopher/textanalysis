import imageio.v2 as imageio
import nltk
import jieba
import numpy as np
from PIL import Image
from wordcloud import ImageColorGenerator, wordcloud
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib

Data = pd.read_csv("风起洛阳.csv", encoding='utf-8', encoding_errors='ignore', header=0)
# header=0 把第一行设置为表头，encoding='gbk'可以读取中文
# 对于不同的csv文件，表头也不一样
comments = Data['text']
comments_list = comments.tolist()
Str = str(comments_list)
# print(comments_list)
# comments_array = np.array(comments_list)


#wbc写的分词和清洗函数
def wash_as(str):
    wordlst = {}
    p1 = re.compile(r"\[.*?\]", re.S)
    list = re.findall(p1, str)
    # 计数1
    for i in list:
        if i not in wordlst:
            wordlst[i] = 1
        else:
            wordlst[i] += 1
    for punct in wordlst.keys():
        str = str.replace(punct, ",")
    jieba.load_userdict('dict.txt')
    seglst = jieba.cut(str)
    # 计数2
    for i in seglst:
        if i not in wordlst:
            wordlst[i] = 1
        else:
            wordlst[i] += 1
    for punct in """！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏.!#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
    """:
        if punct in wordlst:
            wordlst.pop(punct)
    poplst = []
    for char in wordlst.keys():
        if len(char) == 1:
            poplst.append(char)
    for char in poplst:
        wordlst.pop(char)
    # 计数3
    # for char in str:
    #     if ord(char) < 129686 and ord(char) > 127743:
    #         if char not in wordlst:
    #             wordlst[char] = 1
    #         else:
    #             wordlst[char] += 1
    return wordlst


#把洗好的词放入下面的变量里面
text_washed = wash_as(Str)


print(text_washed)

def get_key (dict, value):
    return[k for k, v in dict.items() if v == value]


# 绘制词频直方图
# 主要用意是掌握提取字典里面数据绘图的方法，进而熟悉matplotlib
# 这个可以写成函数吗

# list1 = []
# list2 = []
# for k, v in rank_list:
#     if v > 1:
#         list1.append(k)
#         list2.append(v)
#
# emoji = dict(zip(list1, list2))
#
# fig = plt.figure()
# plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
#
# x1 = list(emoji.keys())
# # x1 = list(map(str, x1))
#
# y1 = list(emoji.values())
#
# plt.bar(x1, y1)
# plt.xlabel("emoji")
# plt.ylabel('词频')
#
# plt.xticks(rotation=90)
# plt.title('emoji词频统计图')
# plt.show()


# 绘制频率直方图的函数
def plt_frequency(Dict={}, xlabel='', ylabel='', title=''):
    rank_list = sorted(Dict.items(), key=lambda d: d[1])

    list1 = []
    list2 = []
    for k, v in rank_list:
        if v > 10:
            list1.append(k)
            list2.append(v)
    re_dict = dict(zip(list1, list2))

    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    x1 = list(re_dict.keys())
    # x1 = list(map(str, x1))

    y1 = list(re_dict.values())

    plt.bar(x1, y1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(rotation=90)
    plt.title(title)
    plt.show()


# 词频图绘制函数
# plt_frequency(text_washed, 'word', 'number', 'frequency')


# 词云生成器
def word_cloud(Dict, picture, output):
    # 蒙版图片路径
    img = imageio.imread(picture)
    w = wordcloud.WordCloud(
        background_color='white',
        mask=img,
        # max_words=300,
        # max_font_size=300,
        # min_font_size=100,
        width=2500,
        height=3000,
        font_path='STZHONGS.TTF',
        mode='RGBA',
        # random_state=1,
        prefer_horizontal=1,
    )
    w.generate_from_frequencies(Dict)
    # 色彩图片路径
    color_source_image = np.array(Image.open(picture))
    colormap = ImageColorGenerator(color_source_image)
    colored_image = w.recolor(color_func=colormap)
    # 保存为图片
    colored_image.to_file(output)
    # w.generate(text)

# word_cloud(text_washed, 'fengqiluoyang.jpg', 'total.png')
