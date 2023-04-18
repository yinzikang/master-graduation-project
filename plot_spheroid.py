#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""基于chatgpt的画椭球的函数

选择好路径，按空格持续输出

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/8/23 11:21 AM   yinzikang      1.0         None
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def update_ellipse(event):
    global current_k_index

    if event.key == ' ':
        # 获取下一个k值
        current_k_index = (current_k_index + 1) % len(k_series)
        k = k_series[current_k_index, :3]
        R = R_series[current_k_index].reshape((3, 3))

        # 将椭球的方程转换为x、y、z坐标系
        A = k * R.transpose()
        X = A.dot(np.vstack((x.flatten(), y.flatten(), z.flatten())))
        X = X.reshape(3, -1).T

        # 清除原始绘图并重新绘制椭球
        ax.clear()
        ax.plot_surface(X[:, 0].reshape(100, 100), X[:, 1].reshape(100, 100),
                        X[:, 2].reshape(100, 100), cmap='jet', rstride=4, cstride=4, alpha=0.5)

        # 将坐标轴尺寸设置为相等大小
        axis_max = 2000
        ax.auto_scale_xyz([-axis_max, axis_max], [-axis_max, axis_max], [-axis_max, axis_max])
        ax.set_title(str(current_k_index) + ' ' + str(k))
        # 显示UI界面
        plt.draw()


# 加载保存的k和direction
path = './eval_results/cabinet surface with plan v7/PPO/04-18-12-48/model/'
k_series = np.load(path + 'K.npy')
R_series = np.load(path + 'direction.npy')

# 用于记录当前使用的k值的索引
current_k_index = 0

# 定义椭球上的角度，计算椭球上每个点的坐标
u = np.linspace(0, 2 * np.pi, 100)  # 偏航
v = np.linspace(0, np.pi, 100)  # 俯仰
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# 创建绘图窗口，并将“空格键”事件与回调函数关联
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.canvas.mpl_connect('key_press_event', update_ellipse)

# 第一次绘制椭球
k = k_series[current_k_index,:3]
R = R_series[current_k_index].reshape((3, 3))
A = k * R.transpose()
X = A.dot(np.vstack((x.flatten(), y.flatten(), z.flatten())))
X = X.reshape(3, -1).T

ax.plot_surface(X[:, 0].reshape(100, 100), X[:, 1].reshape(100, 100),
                X[:, 2].reshape(100, 100), cmap='jet', rstride=4, cstride=4, alpha=0.5)

axis_max = 2000
ax.auto_scale_xyz([-axis_max, axis_max], [-axis_max, axis_max], [-axis_max, axis_max])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(str(current_k_index) + ' ' + str(k))

# 显示绘图
plt.show()
