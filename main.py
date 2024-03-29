from flask import Flask, render_template, request, redirect, Response, jsonify

from process_response import (base64_to_arr, trim_image_array, square_image_array, 
							  resize_image_array, top_n_predictions)
from load_model import load_model
import numpy as np
from keras import backend as K

# Define model type, 'cnn' or 'mlp'
MODEL_TYPE = 'cnn'

app = Flask(__name__)

# Render index.html on server start
@app.route("/")
def output():
	return render_template("index.html")


@app.route('/post_receiver', methods = ['POST'])
def process_post():
	"""
	Recieves POST request from webpage. Request contains base64 encoded 
	input image of a single digit. This method transforms the base64 encoding
	into an image array the right format for input for a model. The prediction
	is sent back to the webpage where it is shown.
	"""

	# Retrieve json from post request
	data = request.get_json(force=True)

	# Base64 to arr
	image_array = base64_to_arr(data)

	# Reshape array to match training data
	image_array = trim_image_array(image_array)
	image_array = square_image_array(image_array)
	image_array = resize_image_array(image_array)

	if MODEL_TYPE == 'mlp':
		# Flatten array for mlp input
		image_array = np.resize(image_array, (image_array.shape[0] * image_array.shape[1],))

		# Load deep mlp model
		model = load_model(json_path="model/deep_mlp.json", h5_path="model/deep_mlp.h5")
	elif MODEL_TYPE == 'cnn':
		# Add third dimension to array for cnn input
		image_array = image_array.reshape(image_array.shape[0], image_array.shape[1], 1)

		# Load cnn model
		model = load_model(json_path="model/cnn.json", h5_path="model/cnn.h5")
	else:
		raise Exception("Unknown model type")

	# Add existing array to new array
	image_array = np.expand_dims(image_array, axis=0)
	
	# Throw array in neural net
	prediction = model.predict(image_array, steps=1)

	# Clear keras session to prepare for futher predictions
	K.clear_session()

	# Get highest predicted probability
	prediction_json = top_n_predictions(prediction, n=1)

	# Return probabilities to webpage
	return jsonify(prediction_json)

if __name__ == '__main__':
	app.run(debug=True)