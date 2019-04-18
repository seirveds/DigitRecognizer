from mnist import MNIST
import numpy as np

def one_hot(labels):
	"""
	One hot encode the mnist Y data
	Input: 
		labels (list/array): non-encoded class labels
	Output: 
		list/array of one-hot encoded class labels
	"""

	# Find amount of classes, length of one-hot vector
	n_classes = len(set(labels))
 
	out = []

	for label in labels:
		one_hot = np.zeros(n_classes)
		# This works because input labels are integers from 0 to 9
		one_hot[label] = 1
		out.append(one_hot)

	return out

def get_train_test(path, train_test_ratio=0.8):
	"""
	Use MNIST library to lead the binary mnist data files
	Input: 
		path (string): path to data files
		train_test_ratio (float): ratio of data used for training
	Output: train and test arrays
	"""
	mndata = MNIST(path)
	mndata.gz = True
	images, labels = mndata.load_training()

	# Encode the labels
	labels = one_hot(labels)

	# Split original training data in train and test
	train_size = int(train_test_ratio * len(images))

	train_images = images[:train_size]
	test_images = images[train_size:]

	train_labels = labels[:train_size]
	test_labels = labels[train_size::]

	return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)