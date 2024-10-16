import numpy as np
import h5py
import matplotlib.pyplot as plt

# 图表格式设置

plt.rcParams['figure.figsize'] = (5.0, 4.0)      # set default size of plots 设置绘图的默认大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 伪随机数种子
np.random.seed(1)


# 激活函数定义区块

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    A = np.maximum(0,Z) 
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s) 
    return dZ





# layer_dims——python数组（列表），包含我们网络中每一层的维度
# 初始化w,b 组件
def initialize_parameters_deep(layer_dims):
    L = len(layer_dims)
    np.random.seed(3)
    parameters = {}
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))    # 易错 np.zeros 只接受一个数据 所以要输入(layer_dims[l],1) 而不是layer_dims[l],1
    return parameters

# 线性前向传播及储存缓存cache
def linear_forward(A, W, b):
    Z = np.dot(W,A) + b
    cache = (A,W,b)
    return Z,cache

# 线性激活  if分类激活方式
def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    if activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A,cache

# 多层线性传播model
def L_model_forward(X, parameters):    # parameters是字典 包含所有初始化过的W，b
    L = len(parameters)//2
    A_prev = X
    caches = []
    for l in range(1,L):
        A_prev = A 
        A,cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation = "relu")
        caches.append(cache)
    AL,cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = 'sigmoid')
    caches.append(cache)
    # 注意cache 储存在caches中是有顺序的 cache包含每一层的 A,W,b 和 z 如cache[0]为第一层的...
    return AL,caches
        

# 计算成本  原盘复制
def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost    
#=====================正向传播完成=======下面开始难点逆向传播