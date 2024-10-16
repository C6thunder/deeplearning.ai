import numpy as np
import matplotlib.pyplot as plt

x_train = np.arange(0, 20, 1)
y_train = x_train**2 + 2


x = x_train**2

# x_train = np.array([5,8,9,13,15])
# y_train = np.array([30,50,60,90,130])
def piandao(x,y,w,b):
    m = x.shape[0]
    dw = 0.0
    db = 0.0
    for i in range(m):
        dw += (w*x[i]+b-y[i])*x[i]
        db += (w*x[i]+b-y[i])
    dw = dw/m
    db = db/m
    return dw,db



def tidu(x,y,w,b,aph,time):
    for i in range(time):
        dw,db = piandao(x,y,w,b)
        w = w - aph*dw
        b = b - aph*db
    return w,b

w = 0.0 ; b = 0.0 ; aph = 1e-5 ; time = 60000
fw,fb = tidu(x,y_train,w,b,aph,time)

print(fw,fb)

plt.plot(x_train,fw*x+fb,linestyle= '--',c='g')
plt.scatter(x_train,y_train)
plt.show()