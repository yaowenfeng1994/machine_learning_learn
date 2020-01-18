# 该文件是草稿文件
import tensorflow as tf

# for p in combinations((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), r=8):
#     print(p)

# tmp_arr = np.array([[1, 3], [2, 1], [3, 1]])
# tmp_arr = np.array([[1, 2, 3], [3, 1, 1]])
# cov_mat = np.cov(tmp_arr)
# print(cov_mat)
# eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# print(eigen_vals)
# print(eigen_vecs)

target_obj = {
    "name": "广东汕头华侨中学",
    "is_deleted": 0,
    "children": [
        {
            "name": "实验楼",
            "is_deleted": 0,
            "children": [
                {
                    "name": "黄鹤楼",
                    "is_deleted": 0,
                    "children": [
                        {
                            "name": "实验楼1",
                            "is_deleted": 0,
                            "children": [
                                {
                                    "name": "实验楼11",
                                    "is_deleted": 0,
                                    "children": [
                                        {
                                            "name": "实验楼2232",
                                            "is_deleted": 0,
                                            "children": [
                                                {
                                                    "name": "实验楼22",
                                                    "is_deleted": 1,
                                                    "children": []
                                                }
                                            ]
                                        }
                                    ]
                                },
                                {
                                    "name": "实验楼22",
                                    "is_deleted": 0,
                                    "children": [
                                        {
                                            "name": "实验楼22",
                                            "is_deleted": 0,
                                            "children": [
                                                {
                                                    "name": "实验楼22",
                                                    "is_deleted": 1,
                                                    "children": []
                                                },
                                                {
                                                    "name": "实验楼33",
                                                    "is_deleted": 0,
                                                    "children": []
                                                },
                                                {
                                                    "name": "实验楼44",
                                                    "is_deleted": 1,
                                                    "children": []
                                                },
                                                {
                                                    "name": "实验楼55",
                                                    "is_deleted": 1,
                                                    "children": []
                                                },
                                                {
                                                    "name": "实验楼66",
                                                    "is_deleted": 0,
                                                    "children": []
                                                },
                                                {
                                                    "name": "实验楼77",
                                                    "is_deleted": 1,
                                                    "children": []
                                                },
                                                {
                                                    "name": "实验楼88",
                                                    "is_deleted": 0,
                                                    "children": []
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                },
                {
                    "name": "实验楼333",
                    "is_deleted": 0,
                    "children": [
                        {
                            "name": "实验楼1",
                            "is_deleted": 0,
                            "children": [
                                {
                                    "name": "实验楼11",
                                    "is_deleted": 1,
                                    "children": []
                                },
                                {
                                    "name": "实验楼22",
                                    "is_deleted": 0,
                                    "children": []
                                }
                            ]
                        }
                    ]
                },
                {
                    "name": "实验楼6677",
                    "is_deleted": 0,
                    "children": [
                        {
                            "name": "实验楼7788",
                            "is_deleted": 1,
                            "children": []
                        }
                    ]
                }
            ]
        }
    ]
}


def traverse_model(obj):
    if obj["is_deleted"] == 1:
        return None
    if "children" in obj and obj["children"].__len__() > 0:
        pop_list = []
        for idx, child_obj in enumerate(obj["children"]):
            if not traverse_model(child_obj):
                pop_list.append(idx)
        for idx, item in enumerate(pop_list):
            if idx == 0:
                obj["children"].pop(item)
            else:
                obj["children"].pop(item - idx)
    return obj


def traverse_model_v2(obj):
    if obj["is_deleted"] == 1:
        return None
    if "children" in obj and obj["children"].__len__() > 0:
        flag = 0
        for _ in range(len(obj["children"])):
            if not traverse_model_v2(obj["children"][flag]):
                # obj["children"].pop(flag)
                obj["children"].pop(flag)
                obj["children"] = obj["children"]
            else:
                flag += 1
    return obj


def remove_deleted_children(org):
    if org.get("is_deleted") == 1:
        return None
    if "children" in org and org.get("children"):
        remove_list = list()
        for index, child in enumerate(org["children"]):
            if child["is_deleted"] == 1:
                remove_list.append(child)
            else:
                org["children"][index] = remove_deleted_children(child)
        for child in remove_list:
            print(child)
            org["children"].remove(child)
        return org
    return org


# result = traverse_model_v2(target_obj)
# print(json.dumps(result))

# 原地循环删除数组
# a = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
# j = 0
# # for i in range(len(a)):
# #     if a[j] == 1:
# #         a.pop(j)
# #     else:
# #         j += 1
# #     print("i: ", i, "  a: ", a, "len: ", len(a))
# # print(a)
#
# b = [1,2,3]
# print(b[:2])

def lengthOfLongestSubstring(s):
    r = dict()
    max_len, left, res = 0, 0, 0
    for idx, letter in enumerate(s):
        if letter in r:
            if r[letter] >= left:
                left = r[letter] + 1
        r[letter] = idx
        max_len = idx - left + 1
        if max_len > res:
            res = max_len

    max_len = len(s) - left
    if max_len > res:
        res = max_len
    return res


def lengthOfLongestSubstringV2(s):
    if not s:
        return 0
    r = list()
    for idx, letter in enumerate(s):
        for i in r:
            if not i["flag"]:
                if letter in i:
                    i["flag"] = True
                else:
                    i[letter] = 1
            else:
                continue
        r.append({"flag": False, letter: 1})
        # max_len = max(len(i) for i in r)
        # print(max_len)
        # print(r)
        # j = 0
        # for i in range(len(r)):
        #     if len(r[j]) < max_len:
        #         r.pop(j)
        #     else:
        #         j += 1

    print(r)
    return max(len(i) for i in r) - 1


if __name__ == "__main__":
    # tf_x = tf.placeholder(tf.float32, shape=[None, 784], name="tf_x")
    # tf_y = tf.placeholder(tf.int32, shape=[None], name="tf_y")

    # shape参数的个数应为维度数，每一个参数的值代表该维度上的长度
    # a = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    # tf_a = tf.reshape(a, shape=[-1, 3, 3, 1])
    # b = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # tf_b = tf.one_hot(indices=b, depth=10)
    # # b = tf.initialize_all_variables()
    # with tf.Session() as sess:
    #     # sess.run(b)
    #     # print(a)
    #     # print(sess.run(tf_a))
    #     # print(tf_a)
    #     print(b)
    #     print(sess.run(tf_b))
    #     print(tf_b)
    # a = {1: {"a": 1, "b": 1, "c": 1}, 2: {"b"}}
    # print(max(len(a[i]) for i in a))

    print(lengthOfLongestSubstringV2("abcavbcbb"))
