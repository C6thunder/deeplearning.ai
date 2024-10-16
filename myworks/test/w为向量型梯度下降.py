import numpy as np 
np.set_printoptions(precision=2)
  
x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])  
y_train = np.array([460, 232, 178])  
  
def dot_product_b(x, w, b):  
    m = np.dot(w, x)  
    p = m + b  
    return p  


def partial_derivative(x, y, w, b):  
    m, n = x.shape  
    dj_dw = np.zeros(n)  
    dj_db = 0.  
  
    for i in range(m):  
        p = dot_product_b(x[i], w, b)  
        err = p - y[i]  
        dj_dw += err * x[i]  
        dj_db += err  
  
    dj_dw = dj_dw / m  
    dj_db = dj_db / m  
  
    return dj_dw, dj_db 

  
def gradient_descent(x, y, w, b, aph, time):  
    for i in range(time):  
        dj_dw, dj_db = partial_derivative(x, y, w, b)  
        w -= aph * dj_dw  
        b -= aph * dj_db  
    return w, b  
  
w = np.zeros(x_train.shape[1])
b = 0.0  
aph = 8e-7 # 学习率  

time = 100000  # 迭代次数  
  
w_f, b_f = gradient_descent(x_train, y_train, w, b, aph, time)  
print(w_f)  
print(f'{b_f:0.2f}')