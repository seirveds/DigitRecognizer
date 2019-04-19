import keras
from load_data import get_train_test

# Load mnist train and test data
x_train, y_train, x_test, y_test = get_train_test(path='mnist_data/')

# Initialize model
model = keras.Sequential()

# Add fully connected layers with some dropout after each layer
model.add(keras.layers.Dense(units=512, activation='relu', input_shape=(28*28,)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(units=256, activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(units=32, activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(units=16, activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(units=10, activation='softmax'))

# Print model stats
model.summary()

# Compile model and ready for fitting
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit model to mnist data
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=30,
                    verbose=1,
                    validation_data=(x_test, y_test))

# Final evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save model structure
model_json = model.to_json()
with open('deep_mlp.json', 'w') as f:
	f.write(model_json)

# Save model weights
model.save_weights('deep_mlp.h5')