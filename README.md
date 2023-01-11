# 大创
带fqly的是写给风起洛阳项目的文件，分别使用了传统的特征工程：word bay，DFIDF和高级特征工程：word2vec  
原始的数据存在风起洛阳微博.csv和风起洛阳豆瓣200短评.csv两个文件里面  
fqly_cluster.csv是当时用DFIDF做的聚类  
fqly_TFIDF.csv是用DFIDF做的特征工程，词向量文件  
fqly_wordbag.csv是用传统词袋法做的特征工程，词向量文件  
weibo-scab.py和Bilibili.py是两个爬虫程序  
fqly_wordbag.py和fqly_w2v.py最开始都是清洗+分词，可以直接用
