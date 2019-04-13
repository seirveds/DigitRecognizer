import keras
from load_data import get_train_test

x_train, y_train, x_test, y_test = get_train_test(path='.')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = keras.Sequential()

model.add(keras.layers.Dense(units=128, activation='relu', input_shape=(28*28,)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(units=10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=30,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
