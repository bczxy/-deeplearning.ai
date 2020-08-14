# -*- coding: utf-8 -*-
import numpy as np
import h5py
import matplotlib.pyplot as plt
from Course1_2.lr_utils import load_dataset
from Util.utils import sigmoid

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
