import keras

def load_model(json_path, h5_path):
	with open(json_path, 'r') as model_file:
		model_json = model_file.read()
	model = keras.models.model_from_json(model_json)
	model.load_weights(h5_path)
	return model

def predict(model, x):
	return model.predict(x)