# 该代码来自 河南大学的 人工智能理论及算法工程研究中心 公众号！！
#原名 主动学习
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from modAL.models import ActiveLearner



X = np.random.choice(np.linspace(0, 20, 10000), size=200, replace=False).reshape(-1, 1)
y = np.cos(X) + np.random.normal(scale=0.3, size=X.shape)  # 使用cos函数

available_styles = plt.style.available
style_to_use = 'seaborn-white' if 'seaborn-white' in available_styles else 'default'

with plt.style.context(style_to_use):
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, c='k', s=20)
    plt.title('cos(x) + noise')
    plt.show()


n_initial = 5
initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False)
X_training, y_training = X[initial_idx], y[initial_idx]



def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]

kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))


regressor = ActiveLearner(
    estimator=GaussianProcessRegressor(kernel=kernel),
    query_strategy=GP_regression_std,
    X_training=X_training, y_training=y_training
)

X_grid = np.linspace(0, 20, 1000).reshape(-1, 1)
y_pred, y_std = regressor.predict(X_grid, return_std=True)
y_pred, y_std = y_pred.ravel(), y_std.ravel()


with plt.style.context(style_to_use):
    plt.figure(figsize=(10, 5))
    plt.plot(X_grid, y_pred, 'b-', label='Prediction')
    plt.fill_between(X_grid.ravel(), y_pred - y_std, y_pred + y_std, alpha=0.2, color='blue', label='Confidence Interval')
    plt.scatter(X, y, c='k', s=20, label='Data')
    plt.title('Initial prediction')
    plt.legend()
    plt.show()



n_queries = 10
for idx in range(n_queries):
    query_idx, query_instance = regressor.query(X)
    regressor.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))



y_pred_final, y_std_final = regressor.predict(X_grid, return_std=True)
y_pred_final, y_std_final = y_pred_final.ravel(), y_std_final.ravel()

with plt.style.context(style_to_use):
    plt.figure(figsize=(10, 8))
    plt.plot(X_grid, y_pred_final, 'b-', label='Prediction')
    plt.fill_between(X_grid.ravel(), y_pred_final - y_std_final, y_pred_final + y_std_final, alpha=0.2, color='blue', label='Confidence Interval')
    plt.scatter(X, y, c='k', s=20, label='Data')
    plt.title('Prediction after active learning')
    plt.legend()
    plt.show()