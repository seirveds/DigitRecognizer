import keras
from math import sqrt
from load_data import get_train_test

# Load mnist test and train data
x_train, y_train, x_test, y_test = get_train_test(path='mnist_data/')

# Get sizes of training and test set
training_size = x_train.shape[0]
test_size = x_test.shape[0]

# Image data is 1d array originally. This needs to be reshaped to a 2d image array
image_sides = int(sqrt(x_train.shape[1])) # Square root of shape[1] assumes square images

# Reshape test and train data to format for for convolutions
x_train = x_train.reshape(training_size, image_sides, image_sides, 1) 
x_test = x_test.reshape(test_size, image_sides, image_sides, 1)

# Get input shape for use in input layer
input_shape = x_train.shape[1:]

# Initialize model
model = keras.Sequential()

# Model structure based on LeNet-5

# 2 convolutional layers
model.add(keras.layers.Conv2D(filters=32,
							  kernel_size=(3,3),
							  input_shape=input_shape,
							  activation='relu'))

model.add(keras.layers.Conv2D(filters=64,
							  kernel_size=(3,3),
							  activation='relu'))

# Max pooling, reduce vector size
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Dropout and flatten 2d vectors to 1d
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())

# Fully connected layer + dropout
model.add(keras.layers.Dense(128, activation='relu'))

model.add(keras.layers.Dropout(0.5))

# Output layer
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

# Compile model for training
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=15,
                    verbose=1,
                    validation_data=(x_test, y_test))

# Final evaluation
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save model structure
model_json = model.to_json()
with open('model/cnn.json', 'w') as f:
	f.write(model_json)

# Save model weights
model.save_weights('model/cnn.h5')