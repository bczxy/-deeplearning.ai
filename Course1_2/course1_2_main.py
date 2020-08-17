# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from Course1_2.lr_utils import load_dataset
from Util import utils

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# show_index = 26
# plt.imshow(train_set_x_orig[show_index])
# plt.show()

print(type(train_set_x_orig))
print(train_set_x_orig.shape)
print(train_set_y.shape)
print(type(train_set_y))

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print(train_set_x_flatten.shape)

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

def plot_model(d):
	# 绘制图
	costs = np.squeeze(d['costs'])
	plt.plot(costs)
	plt.ylabel('cost')
	plt.xlabel('iterations (per hundreds)')
	plt.title("Learning rate =" + str(d["learning_rate"]))
	plt.show()


if __name__ == "__main__":
	# d = utils.logistic_model(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate=0.005, print_cost=True)
	# plot_model(d)

	learning_rates = [0.01, 0.001, 0.0001]
	models = {}
	for i in learning_rates:
		print("learning rate is: " + str(i))
		models[str(i)] = utils.logistic_model(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate=i,
		                                      print_cost=False)
		print('\n' + "-------------------------------------------------------" + '\n')

	for i in learning_rates:
		plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

	plt.ylabel('cost')
	plt.xlabel('iterations')

	legend = plt.legend(loc='upper center', shadow=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.90')
	plt.show()

