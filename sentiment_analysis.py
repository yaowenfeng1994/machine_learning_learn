from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
import warnings
import pyprind

warnings.filterwarnings("ignore")  # 忽略warning


# nltk.download("stopwords")


def preprocessor(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub("[\W]+", " ", text.lower()) + "".join(emoticons).replace("-", "")
    return text


def tokenizer(text):
    stop = stopwords.words("english")
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub("[\W]+", " ", text.lower()) + "".join(emoticons).replace("-", "")
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, "r") as csv:
        next(csv)
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]


# pbar = pyprind.ProgBar(50000)
# print(pbar)
# labels = {"pos": 1, "neg": 0}
#
# df = pd.DataFrame()
# for s in ("test", "train"):
#     for l in ("pos", "neg"):
#         path = "./aclImdb/%s/%s" % (s, l)
#         for file in os.listdir(path):
#             with open(os.path.join(path, file), "r") as infile:
#                 txt = infile.read()
#             df = df.append([[txt, labels[l]]], ignore_index=True)
#             pbar.update()
#
# print(df)
# preprocessor
# np.random.seed(0)
# df = df.reindex(np.random.permutation(df.index))x_train
# df.to_csv("./movie_data.csv", index=False)
# count = CountVectorizer()
# docs = np.array(["The sun is shining", "The weather is sweet", "The sum is shining and the weather is sweet"])
# bag = count.fit_transform(docs)
#
# print(count.vocabulary_)
#
# print(bag.toarray())  # 值是出现了几次，索引是 count.vocabulary_ 的值
#
# tfidf = TfidfTransformer()
# np.set_printoptions(precision=2)
# print(tfidf.fit_transform(bag.toarray()).toarray())


df = pd.read_csv("./movie_data.csv")

# stop = stopwords.words("english")
# print([w for w in tokenizer_porter("a runner likes and runs a lot")[-10:] if w not in stop])
#
# x_train = df.loc[:100, "review"].values
# y_train = df.loc[:100, "sentiment"].values
# x_test = df.loc[100:, "review"].values
# y_test = df.loc[100:, "sentiment"].values
# # print(x_train, y_train, x_test, y_test)
#
# tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
# param_grid = [
#     {"vect__ngram_range": [(1, 1)], "vect__stop_words": [stop, None],
#      "vect__tokenizer": [tokenizer, tokenizer_porter], "clf__penalty": ["l1", "l2"],
#      "clf__C": [1.0, 10.0, 100.0]},
#     {"vect__ngram_range": [(1, 1)], "vect__stop_words": [stop, None],
#      "vect__tokenizer": [tokenizer, tokenizer_porter], "vect__use_idf": [False], "vect__norm": [None],
#      "clf__penalty": ["l1", "l2"], "clf__C": [1.0, 10.0, 100.0]}
# ]
# lr_tfidf = Pipeline([("vect", tfidf), ("clf", LogisticRegression(random_state=0))])
# gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring="accuracy", cv=5, verbose=1, n_jobs=-1)
# gs_lr_tfidf.fit(x_train, y_train)
# print("Best parameter set: %s" % gs_lr_tfidf.best_params_)
# print("CV Accuracy: %.3f" % gs_lr_tfidf.best_score_)
# clf = gs_lr_tfidf.best_estimator_
# print("Test Accuracy: %.3f" % clf.score(x_test, y_test))


# s = stream_docs("./movie_data.csv")
# k = 0
# for i, j in s:
#     print(k, i, j)
#     k += 1
vect = HashingVectorizer(decode_error="ignore", n_features=2**21, preprocessor=None, tokenizer=tokenizer)
clf = SGDClassifier(loss="log", random_state=1, n_iter_no_change=1)
doc_stream = stream_docs("./movie_data.csv")

pbar = pyprind.ProgBar(5)
classes = np.array([0, 1])
for _ in range(5):
    x_train, y_train = get_minibatch(doc_stream, size=1000)
    if not x_train:
        break
    x_train = vect.transform(x_train)
    clf.partial_fit(x_train, y_train, classes=classes)
    pbar.update()

x_test, y_test = get_minibatch(doc_stream, size=1000)
x_test = vect.transform(x_test)
print("Test Accuracy: %.3f" % clf.score(x_test, y_test))
