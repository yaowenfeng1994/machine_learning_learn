from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd
import re


def preprocessor(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub("[\W]+", " ", text.lower()) + "".join(emoticons).replace("-", "")
    return text


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
# df = df.reindex(np.random.permutation(df.index))
# df.to_csv("./movie_data.csv", index=False)


df = pd.read_csv("./movie_data.csv")

t = preprocessor(df.loc[49999, 'review'])
# df["review"] = df["review"].apply(preprocessor)

print(t)

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
