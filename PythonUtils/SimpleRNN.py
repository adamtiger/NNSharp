from keras.models import Sequential
from keras.layers import SimpleRNN
import numpy as np

model = Sequential()
model.add(SimpleRNN(4, activation='relu', stateful=False, batch_input_shape=(4, 3, 3)))
model.compile(optimizer='sgd', loss='mse')

data = np.ndarray((4, 3, 3))
kernel = np.ones((3, 4))
rec_kernel = np.ones((4, 4))
bias = np.array([1.0, -1.0, 2.0, -4.0])

k = 0
for h in range(0, 3):
	for w in range(0, 4):
		k += 1
		kernel[h, w] = k % 5 - 2 

k = 0
for h in range(0, 4):
	for w in range(0, 4):
		k += 1
		rec_kernel[h, w] = k % 5 - 2

parameters = [kernel, rec_kernel, bias]

model.set_weights(parameters)

l = 0
for b in range(0, 4):
	for h in range(0, 3):
		for c in range(0, 3):
			l += 1
			data[b, h, c] = l % 5 + 1

output = model.predict(data, batch_size=4) # the batch_size has no impact on the result here

print(output)


print(model.summary())

print(model.get_config())

print(model.get_weights())
