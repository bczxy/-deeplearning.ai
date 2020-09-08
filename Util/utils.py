# -*- coding: utf-8 -*-
import numpy as np


def sigmoid(x):
	"""
	Compute the sigmoid of x

	Arguments:
	x -- A scalar or numpy array of any size.

	Return:
	s -- sigmoid(x)
	"""
	s = 1 / (1 + np.exp(-x))
	return s


def tanh(x):
	"""
	Compute the tanh of x

	Arguments:
	x -- A scalar or numpy array of any size.

	Return:
	s -- tanh(x)
	"""


def relu(x):
	"""
	Compute the relu of x

	Arguments:
	x -- A scalar or numpy array of any size.

	Return:
	s -- relu(x)
	"""
	s = np.maximum(0, x)
	return s



