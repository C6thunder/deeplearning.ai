import numpy as np  

  

# 假设的原始数组  

arr = np.array([[1, 2], [3, 4]])  

  

# 新行数据  

new_row = np.array([5, 6])  

  

# 添加新行  

new_arr = np.vstack((arr, new_row))  # 使用vstack垂直堆叠  

  

print(new_arr)

# 新列数据  

new_col = np.array([7, 8, 9]).reshape(-1, 1)  # 注意reshape成列向量  

  

# 添加新列  

new_arr = np.hstack((new_arr, new_col))  # 使用hstack水平堆叠  

  

print(new_arr)