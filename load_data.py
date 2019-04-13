from mnist import MNIST
import numpy as np

def one_hot(labels):
	n_classes = len(set(labels))
 
	out = []

	for label in labels:
		one_hot = np.zeros(n_classes)
		one_hot[label] = label
		out.append(one_hot)

	return out

def get_train_test(path):
	mndata = MNIST(path)
	mndata.gz = True
	images, labels = mndata.load_training()

	labels = one_hot(labels)

	# Split original training data in train and test
	train_size = int(0.8 * len(images))

	train_images = images[:train_size]
	test_images = images[train_size:]

	train_labels = labels[:train_size]
	test_labels = labels[train_size::]

	return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)