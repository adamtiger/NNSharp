from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, Dense
import numpy as np

model = Sequential()
model.add(Convolution2D(8, (2, 2), strides=(2, 2), input_shape=(42, 42, 3), activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(2, activation='linear'))

model.compile(optimizer='sgd', loss='mse')

input_data = np.random.randint(0, 255, size=(1, 42, 42, 3))

output_data = model.predict(input_data, 1)

wght = model.get_weights()

model2 = Sequential()
model2.add(Convolution2D(8, (2, 2), strides=(2, 2), input_shape=(42, 42, 3), activation='relu'))
model2.add(Dense(2, activation='linear'))

model2.compile(optimizer='sgd', loss='mse')
model2.set_weights(wght)

output_data2 = model2.predict(input_data, 1)

print(np.array_equal(output_data, output_data2))
print(model.get_config())


