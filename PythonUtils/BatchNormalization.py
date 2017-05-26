from keras.models import Sequential
from keras.layers import BatchNormalization
import numpy as np

model = Sequential()
model.add(BatchNormalization(input_shape=(2, 1, 3)))
model.compile(optimizer='sgd', loss='mse')

data = np.ndarray((4, 2, 1, 3))

l = 0
for b in range(0, 4):
	for h in range(0, 2):
		for w in range(0, 1):
			for c in range(0, 3):
				l += 1
				data[b, h, w, c] = l % 7 - 3

output = model.predict(data, batch_size=1) # the batch_size has no impact on the result here

print(output)
