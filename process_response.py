import base64
import io
from math import floor, ceil
import numpy as np
import re

from PIL import Image, ImageChops, ImageOps


def base64_to_arr(base64_encoding):
	"""
	Converts base64 encoding of image to array.
	Input:
		base64_encoding (string): base64 encoded image received from post request
	Output:
		array containing pixel values of the decoded image
	"""

	# Use regex to extract only the encoding of the image
	# All metadata is discarded
	imgstr = re.search(r'base64,(.*)', base64_encoding).group(1)

	# Transform base64 encoded image to byte representation
	image_bytes = io.BytesIO(base64.b64decode(imgstr))

	# Turn bytes into an PIL Image object
	im = Image.open(image_bytes)

	# Turn Image object into 2d array
	arr = np.array(im)[:,:,0]

	# Invert pixel values to match training data
	# 0 = white, 255 = black
	arr = 255 - arr

	return arr

def trim_image_array(img_arr):
	"""
	Trims all white space around the handwritten digit
	Input:
		img_arr (list/array): array representation of digit image
	Output:
		image array with all extra whitespace trimmed away
	"""
	# Transform image array to Image object
	im = Image.fromarray(img_arr)

	# Create Image object with same size as input, but only white pixels
	background = Image.new(im.mode, im.size, 0)

	# Get the difference between input and background
	difference = ImageChops.difference(im, background)

	# Get bounding box of difference between background and input 
	# This box will contain the drawn number
	bbox = difference.getbbox()

	# Crop the image to the size of the bounding box
	im = im.crop(bbox)

	# Transform cropped image to array
	cropped_img_arr = np.array(im)

	return cropped_img_arr

def square_image_array(img_arr, training_w=28, training_h=28, padding_value=0):
	"""
	Pads an image with some extra padding around the content. If the input
	image is rectangular this function pads the image to be square.
	Input:
		img_arr (array/list): array representation of trimmed digit image
		training_w (int): width of the images used in training
		training_h (int): height of the images used in training
		padding_value (int): value used to pad the image with
	Output:
		Square image array where the length of the sides is a multiple of training_w/h
	"""

	# Get height and with of input image
	h, w = img_arr.shape

	# If the image is already square, pad so that the width and height are a multiple of training_w or training_h
	if h == w:
		padding = training_h - (h % training_h) + training_h
		return np.pad(img_arr, padding, 'constant', constant_values=padding_value)
	# If image is rectangular pad the longest side to a multiple of training_w and pad shortest side
	# to the length of the new padded longest side
	elif h > w:
		# Pad height to a multiple of training height
		# This makes downsizing easier
		total_h_padding = training_h - (h % training_h) + training_h # Add 1 more training h for extra padding

		# If total padding is even add the same amount of padding to both sides
		if total_h_padding % 2 == 0:
			h_padding_top = total_h_padding // 2
			h_padding_bottom = total_h_padding // 2
		# If total padding is odd and 1 pixel more to bottom
		else:
			h_padding_top = floor(total_h_padding / 2)
			h_padding_bottom = ceil(total_h_padding / 2)

		# Get height of image after padding
		new_h = h + total_h_padding

		# Pad w to match new h
		total_w_padding = new_h - w
		# If total padding is even add the same amount of padding to both sides
		if total_h_padding % 2 == 0:
			w_padding_left = total_w_padding // 2
			w_padding_right = total_w_padding // 2
		# If total padding is odd and 1 pixel more to right
		else:
			w_padding_left = floor(total_w_padding / 2)
			w_padding_right = ceil(total_w_padding / 2)

		return np.pad(img_arr, [(h_padding_top,h_padding_bottom), (w_padding_left,w_padding_right)], 
					  'constant', constant_values=padding_value)
	elif h < w:
		# Do above step but with transposed array, transpose back to original orientation after padding
		img_arr_padded = square_image_array(np.transpose(img_arr), training_w=training_w, 
											training_h=training_h, padding_value=padding_value)

		return np.transpose(img_arr_padded)


def resize_image_array(img_arr, w=28, h=28):
	"""
	Resize image array to match input size of neural network
	Input:
		img_arr (array/list): array representation of trimmed and padded digit image
		w (int): width of output image
		h (int): height of output image
	Output:
		Image resized to shape (h,w)
	"""
	# Set argument for thumbnail method
	im_size = h,w

	# Turn image array to Image object
	im = Image.fromarray(img_arr)

	# Downsize Image object to specified size
	im.thumbnail(im_size)

	# Turn Image object back into array
	im_array = np.array(im)

	# thumbnail method forgets a line if pixels sometimes
	# Final check image is desired shape
	if im_array.shape != im_size:
		if im_array.shape[0] < h:
			# Add zeros to bottom
			im_array = np.pad(im_array, [(0,1), (0,0)], 'constant', constant_values=0)
		elif im_array.shape[1] < w:
			# Add zeros to right
			im_array = np.pad(im_array, [(0,0), (0,1)], 'constant', constant_values=0)
		else:
			raise Exception("Image resized to a too large size")
	
	return im_array

def top_n_predictions(prediction, n=1):
	"""
	Return the top n predictions formatted as json, ready to be sent back to webpage
	Input:
		predictions (array): output of keras.model.predict() function
		n (int): return highest n probabilities
	Output:
		json containing the top n predictions in the format [{'class': x, 'prob': y}, ...]
	"""
	# Round predictions and turn into string for json
	prediction = [{'class': class_, 'prob': str(round(prob, 6))} for class_, prob in enumerate(prediction[0])]
	
	# Sort predictions with highest probability on start
	prediction = sorted(prediction, key=lambda x: float(x['prob']), reverse=True)

	return prediction[:n]