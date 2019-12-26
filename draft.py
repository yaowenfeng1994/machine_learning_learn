# 该文件是草稿文件
import numpy as np

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
                    "name": "实验楼1",
                    "is_deleted": 1,
                    "children": []
                },
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
                                    "children": []
                                },
                                {
                                    "name": "实验楼22",
                                    "is_deleted": 1,
                                    "children": []
                                }
                            ]
                        },
                        {
                            "name": "实验楼2",
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
    if "children" in obj:
        if obj["children"].__len__() > 0:
            pop_list = []
            for idx, child_obj in enumerate(obj["children"]):
                if not traverse_model(child_obj):
                    pop_list.append(idx)
            for i in pop_list:
                obj["children"].pop(i)
    return obj


result = traverse_model(target_obj)
print(result)

