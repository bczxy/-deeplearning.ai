# -*- coding: utf-8 -*-
import numpy as np


def sigmoid(z):
	"""
	参数：
		z  - 任何大小的标量或numpy数组。

	返回：
		s  -  sigmoid（z）
	"""
	s = 1 / (1 + np.exp(-z))
	return s


def initialize_with_zeros(dim):
	"""
		此函数为w创建一个维度为（dim，1）的0向量，并将b初始化为0。

		参数：
			dim  - 我们想要的w矢量的大小（或者这种情况下的参数数量）

		返回：
			w  - 维度为（dim，1）的初始化向量。
			b  - 初始化的标量（对应于偏差）
	"""
	w = np.zeros(shape=(dim, 1))
	b = 0
	# 使用断言来确保我要的数据是正确的
	assert (w.shape == (dim, 1))  # w的维度是(dim,1)
	assert (isinstance(b, float) or isinstance(b, int))  # b的类型是float或者是int

	return w, b


def logistic_propagate(w, b, X, Y):
	"""
	    逻辑回归实现前向和后向传播的成本函数及其梯度。
	    参数：
	        w  - 权重，大小不等的数组（dim，1）
	        b  - 偏差，一个标量
	        X  - 矩阵类型为（dim，训练数量）
	        Y  - 真正的“标签”矢量（0或1），矩阵维度为(1,训练数据数量)

	    返回：
	        cost- 逻辑回归的负对数似然成本
	        dw  - 相对于w的损失梯度，因此与w相同的形状
	        db  - 相对于b的损失梯度，因此与b的形状相同
    """
	m = X.shape[1]
	# 正向传播
	A = sigmoid(np.dot(w.T, X) + b)  # 计算激活值。
	cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # 计算成本。

	# 反向传播
	dw = (1 / m) * np.dot(X, (A - Y).T)
	db = (1 / m) * np.sum(A - Y)

	# 使用断言确保我的数据是正确的
	assert (dw.shape == w.shape)
	assert (db.dtype == float)
	cost = np.squeeze(cost)
	assert (cost.shape == ())

	# 创建一个字典，把dw和db保存起来。
	grads = {
		"dw": dw,
		"db": db
	}
	return grads, cost


if __name__ == '__main__':
	print(sigmoid(0))
	pass
