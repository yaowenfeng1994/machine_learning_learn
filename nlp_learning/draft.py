import ssl

from keras.datasets import imdb
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

ssl._create_default_https_context = ssl._create_unverified_context

word_index = imdb.get_word_index()

# num_words=10000的意思是训练集中我们指保留词频最高的前10000个单词。10000名之后的词汇都会被直接忽略，不出现在train_data和test_data中
(x_train, y_train), (x_validation, y_validation) = imdb.load_data(num_words=10000)
print(x_train[0], len(x_train[0]))
# print(type(word_index.items()))
# key值不变，value值加3，并新增了4个键值对
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0  # 用来将每一个sentence扩充到同等长度
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # 未知，可能是生僻单词或是人名
word_index["UNUSED"] = 3
# 将键值对的键与值互换
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# 转译为原句
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# 1是好评，0是差评
# print(decode_review(x_train[1]), y_train[1])

# 合并训练集和评估数据集
# x = np.concatenate((x_train, x_validation), axis=0)
# y = np.concatenate((y_train, y_validation), axis=0)
#
# print('x shape is %s, y shape is %s' % (x.shape, y.shape))
# print('Classes: %s' % np.unique(y))
#
# print('Total words: %s' % len(np.unique(np.hstack(x))))
#
# result = [len(word) for word in x]
# print('Mean: %.2f words (STD: %.2f)' % (np.mean(result), np.std(result)))

# 图表展示 121 122表示整个图形分成一行两列，分别在第一个位置和第二个位置
# plt.subplot(121)
# plt.boxplot(result)
# plt.subplot(122)
# plt.hist(result)
# plt.show()

x_train = sequence.pad_sequences(x_train, padding='post', maxlen=500)
x_validation = sequence.pad_sequences(x_validation, padding='post', maxlen=500)

print(x_train[0], len(x_train[0]))

Embedding(5000, 32, input_length=500)
