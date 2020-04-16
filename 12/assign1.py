"""
Reference:https://blog.csdn.net/y12345678904/article/details/77855936
"""

import numpy as np
import pandas as pd
import re
import random
import jieba
import gensim
from gensim.models import word2vec
from gensim.models import Word2Vec

"""
#preprocessing

#randomly select 100 news
filename = ".\sqlResult_1558435.csv"
content = pd.read_csv(filename,encoding='gb18030')
content_100 = content.sample(n=100,replace=False)
#content_100.to_csv('./content_100.csv',index=0)

#clean and cut text
filename = ".\content_100.csv"
content = pd.read_csv(filename,encoding='utf-8')
content = content['content'].tolist()

#get stopwords list
stop_words = []
with open('./baidu_stopwords.txt',encoding='utf-8') as f:
    for word in f.readlines():
        stop_words.append(word.strip())

def token(string):
    token = ''.join(re.findall(r'[\d|\w]+',str(string)))  #不加str运行content = [token(n) for n in content]报错
    return token

def cut(string):
    return " ".join([w for w in jieba.cut(string) if w not in stop_words]) 

content = [token(n) for n in content]
content = [cut(n) for n in content]

with open('.\content_100_cut.txt','w',encoding="utf-8") as f:
    for a in content:
        f.write(a)

# train word vectors
path = '.\content_100_cut.txt'
sentences = word2vec.LineSentence(path)
model = Word2Vec(sentences, size=100,window=5,min_count=1)
model.save("word2vec.model")
"""

class TextRank(object):
    def __init__(self,word2vec,words,d,window,iternum):
        self.word2vec = word2vec
        self.words = words
        self.d = d
        self.window = window
        self.iternum = iternum
        self.edge_dict = {} #记录节点的边连接字典

    def createNodes(self):
        """
        create the nodes of a graphord2vec,words,d,window,iternum
        """
        tmp_list = []
        for i,word in enumerate(self.words):  #遍历每个词
            if word not in self.edge_dict.keys():
                tmp_list.append(word)
                tmp_set = set()
                left = i-(self.window-1)/2
                right = i + (self.window+1)/2
                if left<0:left=0
                if right>len(self.words):right=len(self.words)
                for j in range(int(left),int(right)):  #遍历每个窗口
                    if j==i:continue
                    tmp_set.add(self.words[j])
                self.edge_dict[word] = tmp_set


    def createMatrix(self):
        """
        create a Matrix representing the weights of edges based on the similarity
        """
        self.matrix = np.zeros([len(set(self.words)), len(set(self.words))])
        self.words_id = list(range(len(self.words)))
        self.word_index = dict(zip(self.words,self.words_id))  # 记录词的index
        self.index_dict = dict(zip(self.words_id,self.words)) # 记录节点index对应的词

        for key in self.edge_dict.keys():
            for w in self.edge_dict[key]:
                self.matrix[self.word_index[key]][self.word_index[w]] = self.word2vec.similarity(key,w)
                self.matrix[self.word_index[w]][self.word_index[key]] = self.word2vec.similarity(key,w)

        # 归一化
        for j in range(self.matrix.shape[1]):
            sum = 0
            for i in range(self.matrix.shape[0]):
                sum += self.matrix[i][j]
            for i in range(self.matrix.shape[0]):
                self.matrix[i][j] /= sum


    def calPR(self):
        self.PR = np.ones([len(self.words), 1])
        for i in range(self.iternum):
            self.PR = (1 - self.d) + self.d * np.dot(self.matrix, self.PR)

    def result(self):
        word_pr={}
        res = dict(zip(self.words,self.PR))
        print(res)
        return res

if __name__ == '__main__':
    model = Word2Vec.load('.\word2vec.model')
    words = [word for word, wv in model.wv.vocab.items()]  # 获取所有词
    tr = TextRank(model,words,d=0.85,window=5,iternum=1)
    tr.createNodes()
    tr.createMatrix()
    tr.calPR()
    tr.result()

