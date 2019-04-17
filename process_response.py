import base64
import io
import re
from PIL import Image, ImageChops, ImageOps
import numpy as np
from math import floor, ceil


def base64_to_arr(base64_encoding):
	imgstr = re.search(r'base64,(.*)', base64_encoding).group(1)

	image_bytes = io.BytesIO(base64.b64decode(imgstr))
	im = Image.open(image_bytes)
	arr = np.array(im)[:,:,0]

	# Invert pixel values to match training data
	# 0 = white, 255 = black
	arr = 255 - arr

	return arr

def trim_image_array(img_arr):
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
	h, w = img_arr.shape

	if h == w:
		padding = training_h - (h % training_h) + training_h
		return np.pad(img_arr, padding, 'constant', constant_values=padding_value)
	elif h > w:
		# Pad height to a multiple of training height
		# This makes downsizing easier
		total_h_padding = training_h - (h % training_h) + training_h # Add 1 more training h for extra padding
		if total_h_padding % 2 == 0:
			h_padding_top = total_h_padding // 2
			h_padding_bottom = total_h_padding // 2
		else:
			h_padding_top = floor(total_h_padding / 2)
			h_padding_bottom = ceil(total_h_padding / 2)

		new_h = h + total_h_padding

		# Pad w to match new h
		total_w_padding = new_h - w
		if total_h_padding % 2 == 0:
			w_padding_left = total_w_padding // 2
			w_padding_right = total_w_padding // 2
		else:
			w_padding_left = floor(total_w_padding / 2)
			w_padding_right = ceil(total_w_padding / 2)

		print(h_padding_top, h_padding_bottom, w_padding_right, w_padding_left)

		return np.pad(img_arr, [(h_padding_top,h_padding_bottom), (w_padding_left,w_padding_right)], 
					  'constant', constant_values=padding_value)
	elif h < w:
		# Do above step but with transposed array, transpose back to original orientation after padding
		img_arr_padded = square_image_array(np.transpose(img_arr), training_w=training_w, 
											training_h=training_h, padding_value=padding_value)

		return np.transpose(img_arr_padded)


def resize_image_array(img_arr, w=28, h=28):

	im_size = w,h

	im = Image.fromarray(img_arr)
	im.thumbnail(im_size)
	return np.array(im)


if __name__ == '__main__':
	img = "data:image/png;base64,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"

	arr = base64_to_arr(img)

	trimmed = trim_image_array(arr)

	trimmed_trans = np.transpose(trimmed)

	squared = square_image_array(trimmed_trans)
	squared2 = square_image_array(trimmed)
	print(squared.shape)

	im = Image.fromarray(squared)
	im.save('test2.png')
	im = Image.fromarray(squared2)
	im.save('test.png')

	resized = resize_image_array(squared)

	print(resized.shape)
	Image.fromarray(resized).save('resize2.png')



