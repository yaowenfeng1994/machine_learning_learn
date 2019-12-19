# 该文件是草稿文件
import numpy as np

# for p in combinations((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), r=8):
#     print(p)

tmp_arr = np.array([[5, 4, 9], [4, 2, 6]])
cov_mat = np.cov(tmp_arr)
print(cov_mat)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print(eigen_vals)
print(eigen_vecs)


