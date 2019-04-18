from flask import Flask, render_template, request, redirect, Response, jsonify

from process_response import base64_to_arr, trim_image_array, square_image_array, resize_image_array, top_n_predictions
from load_model import load_model
import numpy as np
from keras import backend as K

app = Flask(__name__)

@app.route("/")
def output():
	return render_template("index.html")

@app.route('/post_receiver', methods = ['POST'])
def process_post():
	# Retrieve json from post request
	data = request.get_json(force=True)

	# Base64 to arr
	image_array = base64_to_arr(data)

	# Reshape array to match training data
	image_array = trim_image_array(image_array)
	image_array = square_image_array(image_array)
	image_array = resize_image_array(image_array)

	# Flatten array for mlp input
	image_array = np.resize(image_array, (image_array.shape[0] * image_array.shape[1],))

	# Load model
	model = load_model(json_path="model/deep_mlp.json", h5_path="model/deep_mlp.h5")

	# Throw array in neural net
	prediction = model.predict(np.expand_dims(image_array, axis=0), steps=1)

	# Clear keras session to prepare for futher predictions
	K.clear_session()

	# Get highest predicted probability
	prediction_json = top_n_predictions(prediction, n=1)

	print(prediction_json)

	# Return probabilities to webpage
	return jsonify(prediction_json)


if __name__ == '__main__':
	app.run(debug=True)