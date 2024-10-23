import numpy as np  

matrix = np.array([[1, 2, 3],  
                   [4, 5, 6],  
                   [7, 8, 9]])  

# 访问元素  
element = matrix[1, 2]  # 这将返回6，即第二行第三列的元素

print(element)


# 切片  
sub_matrix = matrix[0:2, 1:3]  # 这将返回[[2, 3], [5, 6]]，即前两行和后两列的子矩阵 //左行，右列
print(sub_matrix)

sub_1 = matrix[[0,1,2]]  # 输出0，1，2这三行 
print(sub_1)