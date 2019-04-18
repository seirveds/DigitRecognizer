import keras

def load_model(json_path, h5_path):
	"""
	Loads model structure from json file and model weights from h5 file
	Input:
		json_path (string): path to json file
		h5_path (string): path to h5 file
	Output:
		Keras model object

	"""

	# Load model structure from json file
	with open(json_path, 'r') as model_file:
		model_json = model_file.read()

	model = keras.models.model_from_json(model_json)

	# Load weights from h5 file
	model.load_weights(h5_path)

	return model