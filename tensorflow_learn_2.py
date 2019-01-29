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

pos_file = 'pos.txt'


# 读取文件
def process_file(file):
    with open(file, 'r') as f:
        lex = []
        lines = f.readlines()
        for line in lines:
            words = word_tokenize(line.lower())
            lex += words
        return lex


# 创建词汇表
def create_lexicon(pos_file, neg_file):
    lex_list = list()
    lex_list += process_file(pos_file)
    # lex += process_file(neg_file)
    reduction_lex = [WordNetLemmatizer().lemmatize(word) for word in lex_list]  # 词形还原 (cats->cat)
    # word_count = Counter(reduction_lex)
    # lex = list()
    # {'.': 13944, ',': 10536, 'the': 10120, 'a': 9444, 'and': 7108, 'of': 6624, 'it': 4748, 'to': 3940......}
    # 去掉一些常用词,像the,a and等等，和一些不常用词; 这些词对判断一个评论是正面还是负面没有做任何贡献
    # for word in word_count:
    #     # 这写死了，好像能用百分比
    #     if 20 < word_count[word] < 2000:
    #         print(word, word_count[word])
    #         lex.append(word)
    # return lex

    reduction_lex = list(set(reduction_lex))
    return reduction_lex


# lex:词汇表；review:评论；clf:评论对应的分类，[0,1]代表负面评论 [1,0]代表正面评论
def string_to_vector(lex, review, clf):
    words = word_tokenize(review.lower())
    words = [WordNetLemmatizer().lemmatize(word) for word in words]
    features = np.zeros(len(lex))
    print()
    for word in words:
        if word in lex:
            features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
    return [features, clf]


# 把每条评论转换为向量, 转换原理：
# 假设lex为['woman', 'great', 'feel', 'actually', 'looking', 'latest', 'seen', 'is'] 当然实际上要大的多
# 评论'i think this movie is great' 转换为 [0,1,0,0,0,0,0,1], 把评论中出现的字在lex中标记，出现过的标记为1，其余标记为0
def normalize_dataset(lex):
    dataset = list()
    with open(pos_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [1, 0])  # [array([ 0.,  1.,  0., ...,  0.,  0.,  0.]), [1,0]]
            dataset.append(one_sample)
    # with open(neg_file, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         one_sample = string_to_vector(lex, line, [0, 1])  # [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), [0,1]]]
    #         dataset.append(one_sample)
    # print(len(dataset))
    return dataset


# a = create_lexicon(pos_file, "")
# b = normalize_dataset(a)
# print(b)
# # random.shuffle(b)
# # with open('save.pickle', 'wb') as f:
# #     pickle.dump(b, f)
#
#
# # 取样本中的10%做为测试数据
# test_size = int(len(b) * 0.1)
#
# c = np.array(b)
#
# # train_dataset = c[:-test_size]
# # test_dataset = c[-test_size:]
#
# # Feed-Forward Neural Network
# # 定义每个层有多少'神经元''
n_input_layer = 100  # 输入层

n_layer_1 = 1000  # hide layer
n_layer_2 = 1000  # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层

n_output_layer = 2  # 输出层


# 定义待训练的神经网络
# def neural_network(data):
#     # 定义第一层"神经元"的权重和biases
#     layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
#                    'b_': tf.Variable(tf.random_normal([n_layer_1]))}
#     # 定义第二层"神经元"的权重和biases
#     layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
#                    'b_': tf.Variable(tf.random_normal([n_layer_2]))}
#     # 定义输出层"神经元"的权重和biases
#     layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])),
#                         'b_': tf.Variable(tf.random_normal([n_output_layer]))}
#
#     # w·x+b
#     layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
#     layer_1 = tf.nn.relu(layer_1)  # 激活函数
#     layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
#     layer_2 = tf.nn.relu(layer_2)  # 激活函数
#     layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])
#
#     return layer_output

print(tf.random_normal([n_input_layer, n_layer_1]))

