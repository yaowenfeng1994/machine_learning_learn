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
                    "name": "实验楼",
                    "is_deleted": 0,
                    "children": []
                },
                {
                    "name": "黄鹤楼",
                    "is_deleted": 1,
                    "children": []
                }
            ]
        },
        {
            "name": "罗伊楼",
            "is_deleted": 0,
            "children": [
                {
                    "name": "实验楼",
                    "is_deleted": 1,
                    "children": []
                },
                {
                    "name": "黄鹤楼",
                    "is_deleted": 0,
                    "children": []
                }
            ]
        }
    ]
}


class BeiWeiXiaoHuang(object):

    def traverse_model(self, obj):
        self.remove_model(obj)
        if "children" in obj:
            if obj["children"].__len__() > 0:
                for child_obj in obj["children"]:
                    self.traverse_model(child_obj)
        return obj

    @staticmethod
    def remove_model(remove_obj):
        if remove_obj["is_deleted"] == 1:
            remove_obj.clear()
        return remove_obj


result = BeiWeiXiaoHuang().traverse_model(target_obj)
print(result)

