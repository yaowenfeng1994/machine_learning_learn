#! -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
import os

from nltk.stem import WordNetLemmatizer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# 创建词汇表
# def create_lexicon(pos_file, neg_file):
#     lex = []
#
#     # 读取文件
#     def process_file(file):
#         with open(file, 'r') as f:
#             lex = []
#             lines = f.readlines()
#             # print(lines)
#             for line in lines:
#                 words = word_tokenize(line.lower())
#                 lex += words
#             return lex
#
#     lex += process_file(pos_file)
#     # lex += process_file(neg_file)
#     # print(len(lex))
#     lemmatizer = WordNetLemmatizer()
#     lex = [lemmatizer.lemmatize(word) for word in lex]  # 词形还原 (cats->cat)
#
#     word_count = Counter(lex)
#     # print(word_count)
#     # {'.': 13944, ',': 10536, 'the': 10120, 'a': 9444, 'and': 7108, 'of': 6624, 'it': 4748, 'to': 3940......}
#     # 去掉一些常用词,像the,a and等等，和一些不常用词; 这些词对判断一个评论是正面还是负面没有做任何贡献
#     lex = []
#     for word in word_count:
#         if word_count[word] < 2000 and word_count[word] > 20:  # 这写死了，好像能用百分比
#             lex.append(word)  # 齐普夫定律-使用Python验证文本的Zipf分布 http://blog.topspeedsnail.com/archives/9546
#     return lex

def process_file(file):
    with open(file, 'r') as f:
        lex = []
        lines = f.readlines()

        for line in lines:
            print(line.lower())
            words = word_tokenize(line.lower())
            lex += words
        return lex

print(process_file("pos.txt"))
