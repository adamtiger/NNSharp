import numpy as np
import KerasModeltoJSON as js
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, Dense, Activation, Flatten, MaxPooling1D, MaxPooling2D, AveragePooling1D, \
    AveragePooling2D
from keras.layers import Reshape, Permute, RepeatVector, GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalAveragePooling1D, \
    GlobalAveragePooling2D
from keras.layers import Cropping1D, Cropping2D, BatchNormalization, SimpleRNN, LSTM, GRU
from keras.layers import Dropout, Convolution2D
import json


# json writer
def write(fname, output):
    with open(fname, 'w') as fp:
        shape = output.shape
        if len(shape) == 3:
            temp = np.ndarray((1, shape[0], shape[1], shape[2]))
            temp[0, :, :, :] = output[:, :, :]
            output = temp
        elif len(shape) == 2:
            temp = np.ndarray((1, 1, shape[0], shape[1]))
            temp[0, 0, :, :] = output[:, :]
            output = temp
        output_dict = {"data": output.tolist()}
        json.dump(output_dict, fp)

# data generator
def data_generator(input_size, weight_size):

    input_d = 0
    weight_d = 0
    if not(input_size is None):
        input_d = np.random.randint(0, 255, input_size)

    if not(weight_size is None):
        weight_d = np.random.randint(-5, 5, weight_size)

    return input_d, weight_d


def generate_test_files():

    gen_dropout() # OK

    # CONVOLUTIONS

    # convolution1D tests
    gen_conv_1D_stride_1()  # OK
    gen_conv_1D_stride_2()  # OK

    # convolution2D tests
    gen_conv_2D_stride_1_1()  # OK
    gen_conv_2D_stride_1_2()  # OK

    # cropping1D tests
    gen_cropping1D_tests()  # OK

    # cropping2D tests
    gen_cropping2D_tests()  # OK

    # POOLING

    # avgpooling1D_tests
    gen_avgpool_1D_stride_1()  # OK
    gen_avgpool_1D_stride_2()  # OK

    # avgpooling2D_tests
    gen_avgpool_2D_stride_1_1()  # OK
    gen_avgpool_2D_stride_1_2()  # OK

    # maxpooling1D tests
    gen_maxpool_1D_stride_1()  # OK
    gen_maxpool_1D_stride_2()  # OK

    # maxpooling2D tests
    gen_maxpool_2D_stride_1_1()  # OK
    gen_maxpool_2D_stride_1_2()  # OK

    # globalmaxpooling1D tests
    gen_globalmaxpooling1D()  # OK

    # globalmaxpooling2D tests
    gen_globalmaxpooling2D()  # OK

    # globalaveragepooling1D tests
    gen_globalaveragepooling1D()  # OK

    # globalaveragepooling2D tests
    gen_globalaveragepooling2D()  # OK

    # CORE

    # flatten tests
    gen_flatten()  # OK

    # dense tests
    gen_dense_units_4()  # OK

    # reshape tests
    gen_reshape_tests()  # OK

    # permute tests
    gen_permute_tests()  # OK

    # repeatvector tests
    gen_repeatvector_tests()

    # ACTIVATIONS

    # ELu test
    gen_elu()  # OK

    # HardSigmoid test
    gen_hard_sigmoid()  # OK
	
	# LeakyReLu test
    gen_leakyrelu() 

    # ReLu test
    gen_relu()  # OK

    # Sigmoid test
    gen_sigmoid()  # OK

    # Softmax test
    gen_softmax()  # OK

    # SoftPlus test
    gen_softplus()  # OK

    # SoftSign test
    gen_softsign()  # OK

    # TanH test
    gen_tanh()  # OK

    # NORMALIZATION

    # BatchNormalization
    gen_batchnorm()

    # RNN LAYERS
    # SimpleRNN
    gen_simplernn()

    # LSTM
    gen_lstm()

    # GRU
    gen_gru()


# ---------------------------------------------------------


def gen_dropout():
    model = Sequential()
    model.add(Convolution2D(8, (2, 2), strides=(2, 2), input_shape=(4, 4, 1), activation='relu'))
    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(2, activation='linear'))

    model.compile(optimizer='sgd', loss='mse')

    inp, _ = data_generator((1, 4, 4, 1), None)

    wrt = js.JSONwriter(model, "tests/test_dropout_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_dropout_input.json", inp)
    write("tests/test_dropout_output.json", output)


# CONVOLUTION

def gen_conv_1D_stride_1():
    model = Sequential()

    model.add(Conv1D(3, 2, strides=1, input_shape=(6, 4)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, weight = data_generator((1, 6, 4), (2, 4, 3))

    bias = np.ndarray(3)

    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 2.5

    w = [weight, bias]
    model.set_weights(w)

    wrt = js.JSONwriter(model, "tests/test_conv_1D_1_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_conv_1D_1_input.json", inp)
    write("tests/test_conv_1D_1_output.json", output)


def gen_conv_1D_stride_2():
    model = Sequential()

    model.add(Conv1D(3, 2, strides=2, input_shape=(6, 4)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, weight = data_generator((1, 6, 4), (2, 4, 3))

    bias = np.ndarray(3)

    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 2.5

    w = [weight, bias]
    model.set_weights(w)

    wrt = js.JSONwriter(model, "tests/test_conv_1D_2_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_conv_1D_2_input.json", inp)
    write("tests/test_conv_1D_2_output.json", output)


def gen_conv_2D_stride_1_1():
    model = Sequential()

    model.add(Conv2D(2, (3, 4), strides=(1, 1), input_shape=(4, 5, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, weight = data_generator((1, 4, 5, 2), (3, 4, 2, 2))

    bias = np.ndarray(2)

    bias[0] = 0.5
    bias[1] = 1.5

    w = [weight, bias]
    model.set_weights(w)

    wrt = js.JSONwriter(model, "tests/test_conv_2D_1_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_conv_2D_1_input.json", inp)
    write("tests/test_conv_2D_1_output.json", output)


def gen_conv_2D_stride_1_2():
    model = Sequential()

    model.add(Conv2D(2, (2, 4), strides=(2, 1), input_shape=(4, 5, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, weight = data_generator((1, 4, 5, 2), (2, 4, 2, 2))

    bias = np.ndarray(2)

    bias[0] = 0.5
    bias[1] = 1.5

    w = [weight, bias]
    model.set_weights(w)

    wrt = js.JSONwriter(model, "tests/test_conv_2D_2_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_conv_2D_2_input.json", inp)
    write("tests/test_conv_2D_2_output.json", output)


def gen_cropping1D_tests():
    model = Sequential()

    model.add(Cropping1D(cropping=(1, 2), input_shape=(5, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _ = data_generator((1, 5, 2), None)

    wrt = js.JSONwriter(model, "tests/test_crop_1D_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_crop_1D_input.json", inp)
    write("tests/test_crop_1D_output.json", output)


def gen_cropping2D_tests():
    model = Sequential()

    model.add(Cropping2D(cropping=((1, 1), (1, 2)), input_shape=(4, 5, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _ = data_generator((1, 4, 5, 2), None)

    wrt = js.JSONwriter(model, "tests/test_crop_2D_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_crop_2D_input.json", inp)
    write("tests/test_crop_2D_output.json", output)


# POOLING

# avgpooling1D_tests
def gen_avgpool_1D_stride_1():
    model = Sequential()

    model.add(AveragePooling1D(pool_size=3, strides=1, input_shape=(5, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _ = data_generator((1, 5, 2), None)

    wrt = js.JSONwriter(model, "tests/test_avgpool_1D_1_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_avgpool_1D_1_input.json", inp)
    write("tests/test_avgpool_1D_1_output.json", output)


def gen_avgpool_1D_stride_2():
    model = Sequential()

    model.add(AveragePooling1D(pool_size=3, strides=2, input_shape=(5, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _ = data_generator((1, 5, 2), None)

    wrt = js.JSONwriter(model, "tests/test_avgpool_1D_2_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_avgpool_1D_2_input.json", inp)
    write("tests/test_avgpool_1D_2_output.json", output)


# avgpooling2D_tests
def gen_avgpool_2D_stride_1_1():
    model = Sequential()

    model.add(AveragePooling2D(pool_size=(3, 4), strides=(1, 1), input_shape=(4, 5, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _ = data_generator((1, 4, 5, 2), None)

    wrt = js.JSONwriter(model, "tests/test_avgpool_2D_1_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_avgpool_2D_1_input.json", inp)
    write("tests/test_avgpool_2D_1_output.json", output)


def gen_avgpool_2D_stride_1_2():
    model = Sequential()

    model.add(AveragePooling2D(pool_size=(3, 4), strides=(1, 1), input_shape=(4, 5, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _ = data_generator((1, 4, 5, 2), None)

    wrt = js.JSONwriter(model, "tests/test_avgpool_2D_2_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_avgpool_2D_2_input.json", inp)
    write("tests/test_avgpool_2D_2_output.json", output)


# maxpooling1D tests
def gen_maxpool_1D_stride_1():
    model = Sequential()

    model.add(MaxPooling1D(pool_size=3, strides=1, input_shape=(5, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _ = data_generator((1, 5, 2), None)

    wrt = js.JSONwriter(model, "tests/test_maxpool_1D_1_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_maxpool_1D_1_input.json", inp)
    write("tests/test_maxpool_1D_1_output.json", output)


def gen_maxpool_1D_stride_2():
    model = Sequential()

    model.add(MaxPooling1D(pool_size=3, strides=2, input_shape=(5, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _ = data_generator((1, 5, 2), None)

    wrt = js.JSONwriter(model, "tests/test_maxpool_1D_2_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_maxpool_1D_2_input.json", inp)
    write("tests/test_maxpool_1D_2_output.json", output)


def gen_maxpool_2D_stride_1_1():
    model = Sequential()

    model.add(MaxPooling2D(pool_size=(3, 4), strides=(1, 1), input_shape=(4, 5, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _ = data_generator((1, 4, 5, 2), None)

    wrt = js.JSONwriter(model, "tests/test_maxpool_2D_1_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_maxpool_2D_1_input.json", inp)
    write("tests/test_maxpool_2D_1_output.json", output)


def gen_maxpool_2D_stride_1_2():
    model = Sequential()

    model.add(MaxPooling2D(pool_size=(2, 4), strides=(2, 1), input_shape=(4, 5, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _ = data_generator((1, 4, 5, 2), None)

    wrt = js.JSONwriter(model, "tests/test_maxpool_2D_2_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_maxpool_2D_2_input.json", inp)
    write("tests/test_maxpool_2D_2_output.json", output)


def gen_globalmaxpooling1D():
    model = Sequential()

    model.add(GlobalMaxPooling1D(input_shape=(3, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp = np.ndarray((1, 3, 2))

    inp[0, 0, 0] = 1
    inp[0, 1, 0] = 2
    inp[0, 2, 0] = 0

    inp[0, 0, 1] = 3
    inp[0, 1, 1] = 4
    inp[0, 2, 1] = 0

    wrt = js.JSONwriter(model, "tests/test_globalmaxpool_1D_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_globalmaxpool_1D_input.json", inp)
    write("tests/test_globalmaxpool_1D_output.json", output)


def gen_globalmaxpooling2D():
    model = Sequential()

    model.add(GlobalMaxPooling2D(input_shape=(3, 3, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _ = data_generator((1, 3, 3, 2), None)

    wrt = js.JSONwriter(model, "tests/test_globalmaxpool_2D_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_globalmaxpool_2D_input.json", inp)
    write("tests/test_globalmaxpool_2D_output.json", output)


def gen_globalaveragepooling1D():
    model = Sequential()

    model.add(GlobalAveragePooling1D(input_shape=(3, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp = np.ndarray((1, 3, 2))

    inp[0, 0, 0] = 1
    inp[0, 1, 0] = 2
    inp[0, 2, 0] = 0

    inp[0, 0, 1] = 3
    inp[0, 1, 1] = 4
    inp[0, 2, 1] = 0

    wrt = js.JSONwriter(model, "tests/test_globalavgpool_1D_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_globalavgpool_1D_input.json", inp)
    write("tests/test_globalavgpool_1D_output.json", output)


def gen_globalaveragepooling2D():
    model = Sequential()

    model.add(GlobalAveragePooling2D(input_shape=(3, 3, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _ = data_generator((1, 3, 3, 2), None)

    wrt = js.JSONwriter(model, "tests/test_globalavgpool_2D_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_globalavgpool_2D_input.json", inp)
    write("tests/test_globalavgpool_2D_output.json", output)


# CORE LAYERS

# FLATTEN
def gen_flatten():
    model = Sequential()

    model.add(Flatten(input_shape=(4, 5, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _ = data_generator((1, 4, 5, 2), None)

    wrt = js.JSONwriter(model, "tests/test_flat_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_flat_input.json", inp)
    write("tests/test_flat_output.json", output)


# DENSE
def gen_dense_units_4():
    model = Sequential()

    model.add(Flatten(input_shape=(8, 1, 1)))
    model.add(Dense(4))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, weight = data_generator((1, 8, 1, 1), (8, 4))

    bias = np.ndarray(4)

    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0

    w = [weight, bias]
    model.set_weights(w)

    wrt = js.JSONwriter(model, "tests/test_dense_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_dense_input.json", inp)
    write("tests/test_dense_output.json", output)


def gen_reshape_tests():
    model = Sequential()

    model.add(Reshape((3, 2, 3), input_shape=(3, 3, 2)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _ = data_generator((1, 3, 3, 2), None)

    wrt = js.JSONwriter(model, "tests/test_reshape_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_reshape_input.json", inp)
    write("tests/test_reshape_output.json", output)


def gen_permute_tests():
    model = Sequential()

    model.add(Permute((3, 1, 2), input_shape=(2, 3, 4)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _ = data_generator((1, 2, 3, 4), None)

    wrt = js.JSONwriter(model, "tests/test_permute_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_permute_input.json", inp)
    write("tests/test_permute_output.json", output)


def gen_repeatvector_tests():
    model = Sequential()

    model.add(RepeatVector(3, input_shape=(4,)))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, _  = data_generator((2, 1, 1, 4), None)#np.ndarray((2, 4))

    wrt = js.JSONwriter(model, "tests/test_repeatvector_model.json")
    wrt.save()

    output = np.zeros((2, 1, 3, 4))
    output[:, 0, :, :] = model.predict(inp[:, 0, 0, :], batch_size=1)
    print(output.shape)

    write("tests/test_repeatvector_input.json", inp)
    write("tests/test_repeatvector_output.json", output)


# -------------------------------------------------------------------------------------------------------------

# ACTIVATIONS:

# ELu test
def gen_elu():
    model = Sequential()

    model.add(Flatten(input_shape=(8, 1, 1)))
    model.add(Dense(4))
    model.add(Activation('elu'))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, weight = data_generator((1, 8, 1, 1), (8, 4))

    bias = np.ndarray(4)

    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0

    w = [weight, bias]
    model.set_weights(w)

    wrt = js.JSONwriter(model, "tests/test_elu_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_elu_input.json", inp)
    write("tests/test_elu_output.json", output)


# HardSigmoid test
def gen_hard_sigmoid():
    model = Sequential()

    model.add(Flatten(input_shape=(8, 1, 1)))
    model.add(Dense(4))
    model.add(Activation('hard_sigmoid'))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, weight = data_generator((1, 8, 1, 1), (8, 4))

    bias = np.ndarray(4)

    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0

    w = [weight, bias]
    model.set_weights(w)

    wrt = js.JSONwriter(model, "tests/test_hard_sigmoid_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_hard_sigmoid_input.json", inp)
    write("tests/test_hard_sigmoid_output.json", output)
	
	
# LeakyReLu test
def gen_leakyrelu():
    model = Sequential()

    model.add(Flatten(input_shape=(8, 1, 1)))
    model.add(Dense(4))
    model.add(Activation('leakyrelu'))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, weight = data_generator((1, 8, 1, 1), (8, 4))

    bias = np.ndarray(4)

    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0

    w = [weight, bias]
    model.set_weights(w)

    wrt = js.JSONwriter(model, "tests/test_leakyrelu_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_leakyrelu_input.json", inp)
    write("tests/test_leakyrelu_output.json", output)


# ReLu test
def gen_relu():
    model = Sequential()

    model.add(Flatten(input_shape=(8, 1, 1)))
    model.add(Dense(4))
    model.add(Activation('relu'))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, weight = data_generator((1, 8, 1, 1), (8, 4))

    bias = np.ndarray(4)

    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0

    w = [weight, bias]
    model.set_weights(w)

    wrt = js.JSONwriter(model, "tests/test_relu_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_relu_input.json", inp)
    write("tests/test_relu_output.json", output)


# Sigmoid test
def gen_sigmoid():
    model = Sequential()

    model.add(Flatten(input_shape=(8, 1, 1)))
    model.add(Dense(4))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, weight = data_generator((1, 8, 1, 1), (8, 4))

    bias = np.ndarray(4)

    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0

    w = [weight, bias]
    model.set_weights(w)

    wrt = js.JSONwriter(model, "tests/test_sigmoid_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_sigmoid_input.json", inp)
    write("tests/test_sigmoid_output.json", output)


# Softmax test
def gen_softmax():
    model = Sequential()

    model.add(Flatten(input_shape=(8, 1, 1)))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, weight = data_generator((1, 8, 1, 1), (8, 4))

    bias = np.ndarray(4)

    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0

    w = [weight, bias]
    model.set_weights(w)

    wrt = js.JSONwriter(model, "tests/test_softmax_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_softmax_input.json", inp)
    write("tests/test_softmax_output.json", output)


# SoftPlus test
def gen_softplus():
    model = Sequential()

    model.add(Flatten(input_shape=(8, 1, 1)))
    model.add(Dense(4))
    model.add(Activation('softplus'))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, weight = data_generator((1, 8, 1, 1), (8, 4))

    bias = np.ndarray(4)

    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0

    w = [weight, bias]
    model.set_weights(w)

    wrt = js.JSONwriter(model, "tests/test_softplus_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_softplus_input.json", inp)
    write("tests/test_softplus_output.json", output)


# SoftSign test
def gen_softsign():
    model = Sequential()

    model.add(Flatten(input_shape=(8, 1, 1)))
    model.add(Dense(4))
    model.add(Activation('softsign'))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, weight = data_generator((1, 8, 1, 1), (8, 4))

    bias = np.ndarray(4)

    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0

    w = [weight, bias]
    model.set_weights(w)

    wrt = js.JSONwriter(model, "tests/test_softsign_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_softsign_input.json", inp)
    write("tests/test_softsign_output.json", output)


# TanH test
def gen_tanh():
    model = Sequential()

    model.add(Flatten(input_shape=(8, 1, 1)))
    model.add(Dense(4))
    model.add(Activation('tanh'))
    model.compile(optimizer='rmsprop', loss='mse')

    inp, weight = data_generator((1, 8, 1, 1), (8, 4))

    bias = np.ndarray(4)

    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0

    w = [weight, bias]
    model.set_weights(w)

    wrt = js.JSONwriter(model, "tests/test_tanh_model.json")
    wrt.save()

    output = model.predict(inp, batch_size=1)
    print(output.shape)

    write("tests/test_tanh_input.json", inp)
    write("tests/test_tanh_output.json", output)


# NORMALIZATION

def gen_batchnorm():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(2, 1, 3)))
    model.compile(optimizer='sgd', loss='mse')

    params = [0] * 4

    params[0] = np.array([3, 3, 3])  # gamma
    params[1] = np.array([1, 2, -1])  # beta
    params[2] = np.array([2, 2, 2])  # bias
    params[3] = np.array([5, 5, 5])  # variance

    inp, _ = data_generator((4, 2, 1, 3), None)

    model.set_weights(params)
    output = model.predict(inp, batch_size=1)

    wrt = js.JSONwriter(model, "tests/test_batchnorm_model.json")
    wrt.save()

    print(output.shape)

    write("tests/test_batchnorm_input.json", inp)
    write("tests/test_batchnorm_output.json", output)


# RNN LAYERS
# SimpleRNN
def gen_simplernn():
    model = Sequential()
    model.add(SimpleRNN(4, activation='linear', stateful=False, batch_input_shape=(4, 3, 3)))
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

    output = model.predict(data, batch_size=4)  # the batch_size has no impact on the result here

    wrt = js.JSONwriter(model, "tests/test_simplernn_model.json")
    wrt.save()

    print(output.shape)
    
    inp = np.zeros((4, 1, 3, 3))
    inp[:, 0, :, :] = data[:, :, :]
    ou = np.zeros((4, 1, 1, 4))
    ou[:, 0, 0, :] = output[:, :]
    write("tests/test_simplernn_input.json", inp)
    write("tests/test_simplernn_output.json", ou)


def gen_lstm():
    model = Sequential()
    model.add(LSTM(2, activation='tanh', recurrent_activation='relu', implementation=1, stateful=False,
                   batch_input_shape=(5, 3, 3)))
    model.compile(optimizer='sgd', loss='mse')

    kernel = np.ones((3, 8))
    rec_kernel = np.ones((2, 8))
    bias = np.array([1, 2, -1, 0, 3, 4, 5, -2]) / 10

    k = 0
    for h in range(0, 3):
        for w in range(0, 8):
            k += 1
            kernel[h, w] = (k % 5 - 2) / 10

    k = 0
    for h in range(0, 2):
        for w in range(0, 8):
            k += 1
            rec_kernel[h, w] = (k % 5 - 2) / 10

    parameters = [kernel, rec_kernel, bias]
    model.set_weights(parameters)

    data = np.ndarray((5, 3, 3))

    l = 0
    for b in range(0, 5):
        for h in range(0, 3):
            for c in range(0, 3):
                l += 1
                data[b, h, c] = (l % 5 + 1) / 10

    output = model.predict(data,
                           batch_size=5)  # the batch_size has no impact on the result here # the batch_size has no impact on the result here

    wrt = js.JSONwriter(model, "tests/test_lstm_model.json")
    wrt.save()

    print(output.shape)
    
    inp = np.zeros((5, 1, 3, 3))
    ou = np.zeros((5, 1, 1, 2))
    inp[:, 0, :, :] = data[:, :, :]
    ou[:, 0, 0, :] = output[:, :]
    write("tests/test_lstm_input.json", inp)
    write("tests/test_lstm_output.json", ou)


def gen_gru():
    model = Sequential()
    model.add(GRU(2, activation='tanh', recurrent_activation='relu', implementation=1, stateful=False,
                  batch_input_shape=(5, 3, 3)))
    model.compile(optimizer='sgd', loss='mse')

    kernel = np.ones((3, 6))
    rec_kernel = np.ones((2, 6))
    bias = np.array([1, 2, -1, 0, 3, 4]) / 10

    k = 0
    for h in range(0, 3):
        for w in range(0, 6):
            k += 1
            kernel[h, w] = (k % 5 - 2) / 10

    k = 0
    for h in range(0, 2):
        for w in range(0, 6):
            k += 1
            rec_kernel[h, w] = (k % 5 - 2) / 10

    parameters = [kernel, rec_kernel, bias]
    model.set_weights(parameters)

    data = np.ndarray((5, 3, 3))

    l = 0
    for b in range(0, 5):
        for h in range(0, 3):
            for c in range(0, 3):
                l += 1
                data[b, h, c] = (l % 5 + 1) / 10

    output = model.predict(data,
                           batch_size=5)  # the batch_size has no impact on the result here # the batch_size has no impact on the result here

    wrt = js.JSONwriter(model, "tests/test_gru_model.json")
    wrt.save()

    print(output.shape)

    inp = np.zeros((5, 1, 3, 3))
    ou = np.zeros((5, 1, 1, 2))
    inp[:, 0, :, :] = data[:, :, :]
    ou[:, 0, 0, :] = output[:, :]
    write("tests/test_gru_input.json", inp)
    write("tests/test_gru_output.json", ou)


# Generate ALL the tests:

generate_test_files()
