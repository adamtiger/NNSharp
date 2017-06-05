from keras.models import Sequential
from keras.layers import SimpleRNN
import numpy as np

model = Sequential()
model.add(SimpleRNN(4, activation='tanh', stateful=True, batch_input_shape=(4, 2, 3)))
model.compile(optimizer='sgd', loss='mse')

data = np.ndarray((4, 2, 3))

l = 0
for b in range(0, 4):
	for h in range(0, 2):
		for c in range(0, 3):
			l += 1
			data[b, h, c] = l % 7 - 3

output = model.predict(data, batch_size=4) # the batch_size has no impact on the result here

print(output)


print(model.summary())

print(model.get_config())

print(model.get_weights())
