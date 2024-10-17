import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy   
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
# %matplotlib inline

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

num_px = train_set_x_orig.shape[1]


# 注意：这里假设train_set_x_orig的形状是(m, height, width, channels) 为四维 
## 数据预处理
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
 

## 自己导入的图片数据
k = 4
C = ['3','7','8','9']
C1 = [0,0,1,1]
for i in range(k):
    
    fname = '猫/猫图/cat'+ C[i] +'.jpg'
    image = Image.open(fname)  
    image_resized = image.resize((num_px, num_px))  
    my_image = np.array(image_resized).reshape((1,num_px*num_px*3)).astype(np.float32).T
    train_set_x_flatten = np.hstack((train_set_x_flatten, my_image))
    ep = np.array([[C1[i]]])
    train_set_y = np.hstack((train_set_y,ep))




#标准化
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

### 设计函数
# 初始化函数
def zeros(dim):
    b = 0.
    w = np.zeros((dim,1))
    return w,b
    
# sigm函数
def sigmoid(z):
    fz = 1/(1 + np.exp(-z))
    return fz
    
# dw，db函数
def partial_derivative(X,Y,w,b):
    m = X.shape[1]
    z = np.dot(w.T,X) + b
    A = sigmoid(z)
    dw = np.dot(X,(A-Y).T)/m
    db = np.sum(A-Y)/m
    return dw, db

    
# 梯度下降函数
def gradient_descent(X,Y,num,learnrate):
    w,b = zeros(X.shape[0])
    for i in range(num):
        dw,db = partial_derivative(X,Y,w,b)
        w -= dw*learnrate
        b -= db*learnrate
    return w,b
        
# 0，1划分函数
def divide(X,w,b):
    m = X.shape[1]
    wanty = np.zeros((1,m))
       
    A = sigmoid(np.dot(w.T,X)+b)
    
    for i in range(m):
        if A[0,i]>=0.5:
            wanty[0,i] = 1
        else:
            wanty[0,i] = 0
    return wanty

# 拟合率
def Fitting_rate(X,Y,w,b):
    
    tY = divide(X,w,b)

    rate = 100 - np.mean(np.abs(tY - Y)) * 100
    return rate


## 训练模型
num=3000 ; learnrate=0.3
w,b = gradient_descent(train_set_x,train_set_y,num,learnrate)

## 查看是否过拟合
Difquantity = Fitting_rate(train_set_x,train_set_y,w,b) - Fitting_rate(test_set_x,test_set_y,w,b)
if Difquantity >= 10:
    re = '可能过拟合'
else:
    re = '可能无过拟合'
print(f'该模型粗略错误率:{Difquantity}%\t\t({re})')   



# my_image = input('导入图片名称：')
my_image = input('查看猫图中的：')
    
fname = "猫/猫图/" + my_image + '.jpg'
    
image = Image.open(fname)  
image_resized = image.resize((num_px, num_px))  
my_image = np.array(image_resized).reshape((1,num_px*num_px*3)).astype(np.float32).T



## 对判断样本标准化，否则会使  sigmoid 函数的结果溢出
my_image /= 255 
##

## 判断
plt.imshow(image) # 显示图片

y = divide(my_image,w,b) # 输出0 或 1

if y[0,0] == 1:
    print('有猫')
else:
    print('没有猫')

 
    

