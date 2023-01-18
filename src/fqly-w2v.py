import nltk
import jieba
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import word2vec
from sklearn.manifold import TSNE

plt.rcParams['font.sans-serif'] = ['KaiTi']  # 为了在plt里面正确显示中文
plt.rcParams.update({'font.size': 1})  # 提前设定plt里面字的大小
plt.rcParams['axes.unicode_minus'] = False  # 为了在plt里面正确显示负号

Data = pd.read_csv("风起洛阳微博.csv", encoding='utf-8', encoding_errors='ignore', header=0)
# header=0 把第一行设置为表头，encoding='gbk'可以读取中文
# 对于不同的csv文件，表头也不一样
# 处理重复值
Data = Data.drop_duplicates(['text'], keep='first')
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


# # 从github上找的一个停用词文件
# stopword_file = "stopwords.dat"
# stop_words = read_file(stopword_file)
# print(stop_words)

stop_words = ['的', '了']


# 我写的分词和清洗函数
def normalize_document(doc):
    # pattern = re.compile(r"\[.*?\]", re.S)  # 输入一个re的pattern，交给下面的findall执行
    # Array = re.findall(pattern, doc)  # 返回string中所有与pattern相匹配的全部字串
    # # 返回的形式是数组

    doc = re.sub(r"\d+", '', doc)  # 通过正则表达式删除所有数字
    doc = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", doc)
    doc = re.sub(r"[a-zA-Z]", "", doc)  # 删除所有字母
    jieba.load_userdict('fqly连接词.txt')  # 加载不分开的词

    tokens = jieba.lcut(doc, cut_all=False)  # 分词

    filtered_tokens = [token for token in tokens if token not in stop_words]  # 加载停用词
    # tokens = wpt.tokenize(doc)

    doc = ' '.join(filtered_tokens)
    return doc


# 这里其实有一个问题，如果我们加载停用词的话，很多正常的，比如“不”字就会被去除，导致情感和语句逻辑不通

normalize_doc = np.vectorize(normalize_document)
norm_str = normalize_doc(comments_array)
# print('norm_str:', norm_str)

'''
对于word2vec而言，输入的document需要是类似['sky', 'blue', 'beautiful']这种形式
因此在上文分词+清洗过后，我们需要将每个词分开
采用的是nltk.WordPunctTokenizer()函数
'''
wpt = nltk.WordPunctTokenizer()
tokenized_corpus = [wpt.tokenize(document) for document in norm_str]
# print(tokenized_corpus)

# 进入w2v部分

# 设置参数, 设置参数前我们需要先对data的统计学特征进行一些了解
# 句子长度分布
Data['sentence_length'] = list(map(lambda x: len(x), Data['text']))
# print(Data['sentence_length'])

# # 可视化句子长度分布
# sns.distplot(Data['sentence_length'])
# plt.show()  # 发现大部分document在0-200之间，并且非常显著，超过200后面密度就很小


feature_size = 200
window_context = 5
min_word_count = 1
sample = 1e-4

w2v_model = word2vec.Word2Vec(tokenized_corpus, vector_size=feature_size,
                              sg=1,
                              window=window_context, min_count=min_word_count,
                              sample=sample, epochs=500)

# 用similar words功能测试一下w2v结果
test_similar_words = ['王一博', '风起洛阳', '洛阳', '国产剧']
similar_words = {search_term: [item for item in w2v_model.wv.most_similar([search_term], topn=10)]
                 for search_term in test_similar_words}
print(similar_words)

# similar_words一坨大便，而且完全不收敛
'''
现在similar words终于差不多收敛了，但问题是，很多只出现一两次的句子会对结果产生很大干扰
比如王一博里面的“琪琪”风起洛阳里面的“挺住”
'''


# 通过平均一个document内的word vector找到document vector
def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector


def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in corpus]
    return np.array(features)


# 导出document embedding矩阵
w2v_feature_array = averaged_word_vectorizer(corpus=tokenized_corpus,
                                             model=w2v_model,
                                             num_features=feature_size)
feature_matrix = pd.DataFrame(w2v_feature_array)
feature_matrix.to_csv(r'fqly-w2v-feature-matrix.csv', encoding='utf_8_sig')

# # 用tSNE看下文档聚类
# # tSNE做出来是椭圆形，文档里面好像解释过为什么会出现这种情况
# tsne_words = w2v_model.wv.index_to_key
# # wvs = w2v_model.wv[tsne_words]
# # print(wvs)
#
# tsne_model = TSNE(n_components=2, learning_rate=500, random_state=0, n_iter=5000, perplexity=2)
#
# np.set_printoptions(suppress=True)
# tsne = tsne_model.fit_transform(w2v_feature_array)
#
# w2v_feature_pd = pd.read_csv("fqly-w2v-feature-matrix.csv", encoding='utf-8', encoding_errors='ignore', header=0)
# tsne_labels = w2v_feature_pd.index.tolist()
#
# plt.figure(figsize=(12, 6))
# plt.scatter(tsne[:, 0], tsne[:, 1], c='orange', edgecolors='r')
# for label, x, y in zip(tsne_labels, tsne[:, 0], tsne[:, 1]):
#     plt.annotate(label, xy=(x + 1, y + 1), xytext=(0, 0), textcoords='offset points')
# plt.savefig('tSNEw2v.png', dpi=750, bbox_inches='tight')
