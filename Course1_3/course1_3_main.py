# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from Course1_3.testCases import *
from Course1_3.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from Course1_3.nn_helper import *

np.random.seed(1)

X, Y = load_planar_dataset()

# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
# plt.show()

print("X.shape=" + str(X.shape))
print("Y.shape=" + str(Y.shape))
print("数据集内数据量：" + str(Y.shape[1]))

# 逻辑回归结果（sklearn内置函数实现）
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)
plot_decision_boundary(lambda x: clf.predict(x), X, Y)  # 绘制决策边界
plt.title("Logistic Regression")  # 图标题
LR_predictions = clf.predict(X.T)  # 预测结果
print("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) +
                               np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
      "% " + "(正确标记的数据点所占的百分比)")
# plt.show()

# 使用神经网络
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]  # 隐藏层数量
for i, n_h in enumerate(hidden_layer_sizes):
	plt.subplot(5, 2, i + 1)
	plt.title("Decision Boundary for hidden layer size " + str(n_h))
	parameters = nn_model(X, Y, n_h, num_iterations=10000, learning_rate=0.5)
	plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
	predictions = predict(parameters, X)
	accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
	print("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))
plt.show()

