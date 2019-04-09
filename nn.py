import numpy as np

class NeuralNet():

	def __init__(self, layer_sizes):
		weight_shapes = [(h,w) for h,w in zip(layer_sizes[1:], layer_sizes[:-1])]

		self.weight_array = [np.random.normal(size=shape)/shape[1]**.5 for shape in weight_shapes]

		self.biases = [np.zeros((shape,1)) for shape in layer_sizes[1:]]


	def feed_forward(self, a): 
		for weights, biases in zip(self.weight_array, self.biases):
			a = self.activation(np.matmul(weights, a) + biases)
		return a

	@staticmethod
	def activation(x):
		return 1 / (1+np.exp(-x))


if __name__ == '__main__':
	layer_sizes = (3,5,10)
	x = np.ones((layer_sizes[0],1))

	nn = NeuralNet(layer_sizes)

	p = nn.feed_forward(x)

	print(p)