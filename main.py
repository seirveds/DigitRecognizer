from flask import Flask, render_template, request, redirect, Response

from process_response import base64_to_arr, trim_image_array, square_image_array, resize_image_array
from predict import load_model
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
	print("Data recieved")


	# Base64 to arr
	image_array = base64_to_arr(data)
	print("Data decoded")



	# Reshape array to match training data
	image_array = trim_image_array(image_array)
	print('Data trimmed')

	image_array = square_image_array(image_array)
	print("Data squared")


	image_array = resize_image_array(image_array)
	print("Data resized")



	# Flatten array for mlp input
	image_array = np.resize(image_array, (image_array.shape[0] * image_array.shape[1],))
	print("Data flattened")

	# Load model
	model = load_model(json_path="model/deep_mlp.json", h5_path="model/deep_mlp.h5")
	print("Model loaded")

	print(image_array.shape)

	# model._make_predict_function()


	# Throw array in neural net
	prediction = model.predict( np.expand_dims(image_array, axis=0), steps=1)
	print("Prediction made")

	# Round predictions
	prediction = [p for p in prediction[0]]
	for i,j in enumerate(prediction):
		print("class: {}\tprobability: {}".format(i,round(j, 5))) 

	# Clear keras session
	K.clear_session()	

	# Return probabilities to website

	return 'True'


if __name__ == '__main__':
	app.run(debug=True)