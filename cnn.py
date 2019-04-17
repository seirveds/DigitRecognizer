import keras
from math import sqrt
from load_data import get_train_test

x_train, y_train, x_test, y_test = get_train_test(path='.')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

training_size = x_train.shape[0]
test_size = x_test.shape[0]

image_sides = int(sqrt(x_train.shape[1])) # Square root of shape[1] assumes square images

x_train = x_train.reshape(training_size, image_sides, image_sides, 1) 
x_test = x_test.reshape(test_size, image_sides, image_sides, 1)

input_shape = x_train.shape[1:]


model = keras.Sequential()

model.add(keras.layers.Conv2D(filters=32,
							  kernel_size=(3,3),
							  input_shape=input_shape,
							  activation='relu'))

# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(filters=64,
							  kernel_size=(3,3),
							  activation='relu'))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))

model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(10, activation='softmax'))




model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=15,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

