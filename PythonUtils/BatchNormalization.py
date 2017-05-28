from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D
import numpy as np

model = Sequential()
model.add(BatchNormalization(input_shape=(2, 1, 3)))
model.add(Conv2D(2, (1, 1)))
model.compile(optimizer='sgd', loss='mse')

params = [0] * 4 

params[0] = np.array([3,3,3]) # gamma
params[1] = np.array([1,2,-1]) # beta
params[2] = np.array([2,2,2]) # bias
params[3] = np.array([5,5,5]) # variance

data = np.ndarray((4, 2, 1, 3))

l = 0
for b in range(0, 4):
	for h in range(0, 2):
		for w in range(0, 1):
			for c in range(0, 3):
				l += 1
				data[b, h, w, c] = l % 7 - 3

model.set_weights(params)
output = model.predict(data, batch_size=1) # the batch_size has no impact on the result here

print(output)


print(model.summary())

print(model.get_config())

print(model.get_weights())

