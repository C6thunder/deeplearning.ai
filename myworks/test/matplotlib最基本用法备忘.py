import matplotlib.pyplot as plt  
import numpy as np  
  
# 生成x数据  
x = np.linspace(-10, 10, 400)  
# 计算对应的y值  
y = x**2  
  
# 绘制曲线  
plt.plot(x, y)  
plt.title('Parabola')  # 添加标题  
plt.xlabel('x')  # x轴标签  
plt.ylabel('y')  # y轴标签  
plt.grid(True)  # 显示网格  
plt.show()  # 显示图形