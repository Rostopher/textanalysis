import nltk
import jieba
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

Data = pd.read_csv("风起洛阳微博.csv", encoding='utf-8', encoding_errors='ignore', header=0)
# header=0 把第一行设置为表头，encoding='gbk'可以读取中文
# 对于不同的csv文件，表头也不一样
comments = Data['text']
comments_list = comments.tolist()
# print(comments_list)
comments_array = np.array(comments_list)

'''
这里就用我写的分词函数了，wbc的那个大杂烩放在feipan的分析文件里
我的分词逻辑：输入的是array，保证每个输出document的独立性
输出结果是[434 rows x 3789 columns]矩阵，和xlsx文件里的总评论数匹配
需要改进：jieba的中文停用词没有删除，数字没有删除，
特殊符号因为remove函数的原因，只删除了第一次出现的

第二个doc = re.sub()中每个unicode编码：
\u4e00-\u9fa5                                      汉字的unicode范围
\u0030-\u0039                                     数字的unicode范围
\u0041-\u005a                                     大写字母unicode范围
\u0061-\u007a                                     小写字母unicode范围
'''


# 按行读取文件，返回文件的行字符串列表
def read_file(file_name):
    fp = open(file_name, "r", encoding="utf-8")
    content_lines = fp.readlines()
    fp.close()
    # 去除行末的换行符，否则会在停用词匹配的过程中产生干扰
    for i in range(len(content_lines)):
        content_lines[i] = content_lines[i].rstrip("\n")
    return content_lines


# 从github上找的一个停用词文件
stopword_file = "stopwords.dat"
stop_words = read_file(stopword_file)


# 我写的分词和清洗函数
def normalize_document(doc):
    # pattern = re.compile(r"\[.*?\]", re.S)  # 输入一个re的pattern，交给下面的findall执行
    # Array = re.findall(pattern, doc)  # 返回string中所有与pattern相匹配的全部字串
    # # 返回的形式是数组

    doc = re.sub(r"\d+", '', doc)  # 通过正则表达式删除所有数字
    doc = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", doc)
    jieba.load_userdict('fqly.txt')  # 加载不分开的词

    tokens = jieba.lcut(doc, cut_all=False)  # 分词

    filtered_tokens = [token for token in tokens if token not in stop_words]  # 加载停用词
    # tokens = wpt.tokenize(doc)

    doc = ' '.join(filtered_tokens)
    return doc


normalize_doc = np.vectorize(normalize_document)
norm_str = normalize_doc(comments_array)
print('norm_str:', norm_str)

# 构建word bag model矩阵
cv = CountVectorizer(min_df=0., max_df=1.)  # 这是一个很强大的函数，好像还集合了预处理
cv_matrix = cv.fit_transform(norm_str)
cv_matrix = cv_matrix.toarray()
vocab = cv.get_feature_names()  # 这里是获得表头，所有不一样的词
WordBag_matrix = pd.DataFrame(cv_matrix, columns=vocab)
print(WordBag_matrix)
WordBag_matrix.to_csv(r'F:\code\中期检查\fqly_word_bag.csv', encoding='utf_8_sig')

# 构建TFIDF model矩阵
tt = TfidfTransformer(norm='l2', use_idf=True)  # use_idf=True说明使用idf
tt_matrix = tt.fit_transform(cv_matrix)  # 将前面得到的词袋矩阵转化成TF-IDF得分矩阵

# 可视化
tt_matrix = tt_matrix.toarray()
vocab = cv.get_feature_names()
TFIDF_matrix = pd.DataFrame(np.round(tt_matrix, 2), columns=vocab)
TFIDF_matrix.to_csv(r'F:\code\中期检查\fqly_TFIDF.csv', encoding='utf_8_sig')


# 构建相似度矩阵
# 基于TFIDF特征矩阵构建文档余弦相似度矩阵
similarity_matrix = cosine_similarity(tt_matrix)
similarity_df = pd.DataFrame(similarity_matrix)

# 层次聚类
Z = linkage(similarity_matrix, 'ward')
# 注意这里输入的并不是Dataframe版本的矩阵，而是直接用cosine_similarity构建出来的
pd.DataFrame(Z,
             columns=['Document\Cluster 1', 'Document\Cluster 2', 'Distance', 'Cluster Size'],
             dtype='object')

# 可视化聚类过程
plt.figure(figsize=(8, 3))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
dendrogram(Z)
plt.axhline(y=4.0, c='k', ls='--', lw=0.5)  # 这是画线的一行，ls=line style, lw=line weight

plt.savefig('visualize_cluster.png', dpi=750, bbox_inches='tight')

# 选取聚类criteria
max_dist = 4.0

# 展示criteria下聚类结果
comments_df = pd.DataFrame({'Document': comments_array})
cluster_labels = fcluster(Z, max_dist, criterion='distance')
cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])  # 聚类结果的展示，哪个文档被归到哪个类
clusterlabel_comments = pd.concat([comments_df, cluster_labels], axis=1)
# 这个函数是合并dataframe或series的，axis=1说明是一列一列排
cluster_labels.to_csv(r'F:\code\中期检查\fqly_cluster.csv', encoding='utf_8_sig')
clusterlabel_comments.to_csv(r'F:\code\中期检查\fqly_cluster_with_comments.csv', encoding='utf_8_sig')