# -*- coding: utf-8 -*-
import numpy as np
from Course1_3.planar_utils import sigmoid


def layer_size(X, Y):
	"""
    参数：
     X - 输入数据集,维度为（输入的数量，训练/测试的数量）
     Y - 标签，维度为（输出的数量，训练/测试数量）

    返回：
     n_x - 输入层的数量
     n_h - 隐藏层的数量
     n_y - 输出层的数量
    """
	n_x = X.shape[0]
	n_h = 4  # 这里使用的隐藏层是4，hard code
	n_y = Y.shape[0]
	return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
	"""
	参数：
		n_x - 输入层节点的数量
		n_h - 隐藏层节点的数量
		n_y - 输出层节点的数量

	返回：
		parameters - 包含参数的字典：
			W1 - 权重矩阵,维度为（n_h，n_x）
			b1 - 偏向量，维度为（n_h，1）
			W2 - 权重矩阵，维度为（n_y，n_h）
			b2 - 偏向量，维度为（n_y，1）
	"""
	np.random.seed(2)
	W1 = np.random.randn(n_h, n_x) * 0.01  # 这里提前进行了转置
	b1 = np.zeros(shape=(n_h, 1))
	W2 = np.random.randn(n_y, n_h) * 0.01
	b2 = np.zeros(shape=(n_y, 1))

	assert (W1.shape == (n_h, n_x))
	assert (b1.shape == (n_h, 1))
	assert (W2.shape == (n_y, n_h))
	assert (b2.shape == (n_y, 1))

	parameters = {"W1": W1,
	              "b1": b1,
	              "W2": W2,
	              "b2": b2}

	return parameters


def forward_propagation(X, parameters):
	"""
    参数：
         X - 维度为（n_x，m）的输入数据。
         parameters - 初始化函数（initialize_parameters）的输出

    返回：
         A2 - 使用sigmoid()函数计算的第二次激活后的数值
         cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型变量
    """
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	# 前向传播
	Z1 = np.dot(W1, X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2, A1) + b2
	A2 = sigmoid(Z2)
	assert (A2.shape == (1, X.shape[1]))
	cache = {"Z1": Z1,
	         "A1": A1,
	         "Z2": Z2,
	         "A2": A2}

	return A2, cache


def compute_cost(A2, Y):
	"""
    计算交叉熵成本，

    参数：
         A2 - 使用sigmoid()函数计算的第二次激活后的数值
         Y - "True"标签向量,维度为（1，数量）

    返回：
         成本 - 交叉熵成本
    """
	m = Y.shape[1]
	logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
	cost = - np.sum(logprobs) / m
	cost = float(np.squeeze(cost))
	return cost


def backward_propagation(parameters, cache, X, Y):
	"""
    参数：
     parameters - 包含我们的参数的一个字典类型的变量。
     cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型的变量。
     X - 输入数据，维度为（2，数量）
     Y - “True”标签，维度为（1，数量）

    返回：
     grads - 包含W和b的导数一个字典类型的变量。
    """
	m = X.shape[1]
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	A1 = cache["A1"]
	A2 = cache["A2"]

	dZ2 = A2 - Y  # sigmoid函数的导数
	dW2 = np.dot(dZ2, A1.T) / m
	db2 = np.sum(dZ2, axis=1, keepdims=True) / m
	dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))  # tanh函数的导数是 1- x * x
	dW1 = np.dot(dZ1, X.T) / m
	db1 = np.sum(dZ1, axis=1, keepdims=True) / m
	grads = {"dW1": dW1,
	         "db1": db1,
	         "dW2": dW2,
	         "db2": db2}

	return grads


def update_parameters(parameters, grads, learning_rate=1.2):
	"""
    参数：
     parameters - 包含参数的字典类型的变量。
     grads - 包含导数值的字典类型的变量。
     learning_rate - 学习速率

    返回：
     parameters - 包含更新参数的字典类型的变量。
    """
	W1, W2 = parameters["W1"], parameters["W2"]
	b1, b2 = parameters["b1"], parameters["b2"]

	dW1, dW2 = grads["dW1"], grads["dW2"]
	db1, db2 = grads["db1"], grads["db2"]

	W1 = W1 - learning_rate * dW1
	b1 = b1 - learning_rate * db1
	W2 = W2 - learning_rate * dW2
	b2 = b2 - learning_rate * db2

	parameters = {"W1": W1,
	              "b1": b1,
	              "W2": W2,
	              "b2": b2}

	return parameters


def nn_model(X, Y, n_h, num_iterations, learning_rate=0.5, print_cost=False):
	"""
    参数：
        X - 数据集,维度为（2，示例数）
        Y - 标签，维度为（1，示例数）
        n_h - 隐藏层的数量
        num_iterations - 梯度下降循环中的迭代次数
        learning_rate - 学习率
        print_cost - 如果为True，则每1000次迭代打印一次成本数值

    返回：
        parameters - 模型学习的参数，它们可以用来进行预测。
    """
	np.random.seed(3)
	n_x, _, n_y = layer_size(X, Y)
	parameters = initialize_parameters(n_x, n_h, n_y)

	for i in range(num_iterations):
		A2, cache = forward_propagation(X, parameters)
		cost = compute_cost(A2, Y)
		grads = backward_propagation(parameters, cache, X, Y)
		parameters = update_parameters(parameters, grads, learning_rate)

		if print_cost:
			if i % 1000 == 0:
				print("第 ", i, " 次循环，成本为：" + str(cost))
	return parameters

def predict(parameters, X):
	"""
	使用model返回的parameters进行预测
	参数：
		parameters - 包含参数的字典类型的变量。
	    X - 输入数据（n_x，m）

    返回
		predictions - 我们模型预测的向量（红色：0 /蓝色：1）
	"""
	A2, cache = forward_propagation(X, parameters)
	predictions = np.round(A2)  # 以0.5为界区分2分类

	return predictions
