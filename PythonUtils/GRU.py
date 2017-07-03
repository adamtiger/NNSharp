from keras.models import Sequential
from keras.layers import GRU
import numpy as np

model = Sequential()
ly = GRU(2, activation='tanh', recurrent_activation='relu',implementation = 1, stateful=False, batch_input_shape=(5, 3, 3))
model.add(ly)
model.compile(optimizer='sgd', loss='mse')

kernel = np.ones((3, 6))
rec_kernel = np.ones((2, 6))
bias = np.array([1, 2, -1, 0, 3, 4])/10

k = 0
for h in range(0, 3):
	for w in range(0, 6):
		k += 1
		kernel[h, w] = (k % 5 - 2)/10


k = 0
for h in range(0, 2):
	for w in range(0, 6):
		k += 1
		rec_kernel[h, w] = (k % 5 - 2)/10


parameters = [kernel, rec_kernel, bias]
model.set_weights(parameters)

data = np.ndarray((5, 3, 3))

l = 0
for b in range(0, 5):
	for h in range(0, 3):
		for c in range(0, 3):
			l += 1
			data[b, h, c] = (l % 5 + 1)/10


output = model.predict(data, batch_size=5) # the batch_size has no impact on the result here

print(output)


print(model.summary())

print(model.get_config())

print(model.get_weights())
