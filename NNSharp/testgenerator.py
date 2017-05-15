import numpy as np
import KerasModeltoJSON as js
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, Dense, Activation, Flatten, MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D
import json

# json writer
def write(fname, output):
    with open(fname, 'w') as fp:
        json.dump(output, fp)

def generate_test_files():

    # CONVOLUTIONS

    # convolution1D tests
    gen_conv_1D_stride_1()
    gen_conv_1D_stride_2()

    # convolution2D tests
    gen_conv_2D_stride_1_1()
    gen_conv_2D_stride_1_2()

    # POOLING

    # avgpooling1D_tests
    gen_avgpool_1D_stride_1()
    gen_avgpool_1D_stride_2()

    # avgpooling2D_tests
    gen_avgpool_2D_stride_1_1()
    gen_avgpool_2D_stride_1_2()

    # maxpooling1D tests
    gen_maxpool_1D_stride_1()
    gen_maxpool_1D_stride_2()

    # maxpooling2D tests
    gen_maxpool_2D_stride_1_1()
    gen_maxpool_2D_stride_1_2()

    # flatten tests
    gen_flatten()

    # dense tests
    gen_dense_units_4()

    # ACTIVATIONS

    # ELu test
    gen_elu()

    # HardSigmoid test
    gen_hard_sigmoid()

    # ReLu test
    gen_relu()

    # Sigmoid test
    gen_sigmoid()

    # Softmax test
    gen_softmax()

    # SoftPlus test
    gen_softplus()

    # SoftSign test
    gen_softsign()

    # TanH test
    gen_tanh()

# ---------------------------------------------------------

# CONVOLUTION

def gen_conv_1D_stride_1():
    model = Sequential()
    
    model.add(Conv1D(3, 2, strides=1, input_shape=(6,4)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    weight = np.ndarray((2,4,3))
    
    weight[0,0,0] = 0 
    weight[0,0,1] = 1.5
    weight[0,0,2] = 2 
    
    weight[0,1,0] = 0.5
    weight[0,1,1] = -1 
    weight[0,1,2] = -2
    
    weight[0,2,0] = 3 
    weight[0,2,1] = 0
    weight[0,2,2] = 1
     
    weight[0,3,0] = 1
    weight[0,3,1] = -3 
    weight[0,3,2] = 2.5
    
    weight[1,0,0] = 1.5 
    weight[1,0,1] = 0.5
    weight[1,0,2] = -2 
    
    weight[1,1,0] = 1.5
    weight[1,1,1] = -0.5
    weight[1,1,2] = 2.5
    
    weight[1,2,0] = 2.5 
    weight[1,2,1] = 0.5
    weight[1,2,2] = -1.5 
    
    weight[1,3,0] = -1
    weight[1,3,1] = 3 
    weight[1,3,2] = 0.5
    
    bias = np.ndarray(3)
    
    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 2.5
    
    w = [weight, bias]
    model.set_weights(w)
    
    inp = np.ndarray((1,6,4))
    
    inp[0,0,0] = 0 
    inp[0,0,1] = 1
    inp[0,0,2] = 2
    inp[0,0,3] = 1.5
     
    inp[0,1,0] = 1
    inp[0,1,1] = 0 
    inp[0,1,2] = 0
    inp[0,1,3] = 0.6
    
    inp[0,2,0] = 2 
    inp[0,2,1] = 1
    inp[0,2,2] = 2 
    inp[0,2,3] = 2.5
    
    inp[0,3,0] = 1
    inp[0,3,1] = 0 
    inp[0,3,2] = -1
    inp[0,3,3] = 0
    
    inp[0,4,0] = 1 
    inp[0,4,1] = -2
    inp[0,4,2] = 3 
    inp[0,4,3] = 3.5
    
    inp[0,5,0] = 2 
    inp[0,5,1] = 1
    inp[0,5,2] = 4 
    inp[0,5,3] = 3.5
    
    wrt = js.JSONwriter(model, "tests/test_conv_1D_1_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_conv_1D_1_output.json", output.tolist())

def gen_conv_1D_stride_2():
    model = Sequential()
    
    model.add(Conv1D(3, 2, strides=2, input_shape=(6,4)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    weight = np.ndarray((2,4,3))
    
    weight[0,0,0] = 0 
    weight[0,0,1] = 1.5
    weight[0,0,2] = 2 
    
    weight[0,1,0] = 0.5
    weight[0,1,1] = -1 
    weight[0,1,2] = -2
    
    weight[0,2,0] = 3 
    weight[0,2,1] = 0
    weight[0,2,2] = 1
     
    weight[0,3,0] = 1
    weight[0,3,1] = -3 
    weight[0,3,2] = 2.5
    
    weight[1,0,0] = 1.5 
    weight[1,0,1] = 0.5
    weight[1,0,2] = -2 
    
    weight[1,1,0] = 1.5
    weight[1,1,1] = -0.5
    weight[1,1,2] = 2.5
    
    weight[1,2,0] = 2.5 
    weight[1,2,1] = 0.5
    weight[1,2,2] = -1.5 
    
    weight[1,3,0] = -1
    weight[1,3,1] = 3 
    weight[1,3,2] = 0.5
    
    bias = np.ndarray(3)
    
    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 2.5
    
    w = [weight, bias]
    model.set_weights(w)
    
    inp = np.ndarray((1,6,4))
    
    inp[0,0,0] = 0 
    inp[0,0,1] = 1
    inp[0,0,2] = 2
    inp[0,0,3] = 1.5
     
    inp[0,1,0] = 1
    inp[0,1,1] = 0 
    inp[0,1,2] = 0
    inp[0,1,3] = 0.6
    
    inp[0,2,0] = 2 
    inp[0,2,1] = 1
    inp[0,2,2] = 2 
    inp[0,2,3] = 2.5
    
    inp[0,3,0] = 1
    inp[0,3,1] = 0 
    inp[0,3,2] = -1
    inp[0,3,3] = 0
    
    inp[0,4,0] = 1 
    inp[0,4,1] = -2
    inp[0,4,2] = 3 
    inp[0,4,3] = 3.5
    
    inp[0,5,0] = 2 
    inp[0,5,1] = 1
    inp[0,5,2] = 4 
    inp[0,5,3] = 3.5
    
    wrt = js.JSONwriter(model, "tests/test_conv_1D_2_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_conv_1D_2_output.json", output.tolist())

def gen_conv_2D_stride_1_1():
    model = Sequential()
    
    model.add(Conv2D(2, (3,4), strides=(1,1), input_shape=(4,5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    weight = np.ndarray((3,4,2,2))
    
    weight[0,0,0,0] = 0 
    weight[0,0,0,1] = 1.5
    weight[0,0,1,0] = 2 
    weight[0,0,1,1] = 0.5
    
    weight[0,1,0,0] = -1 
    weight[0,1,0,1] = -2
    weight[0,1,1,0] = 3 
    weight[0,1,1,1] = 0
    
    weight[0,2,0,0] = 1 
    weight[0,2,0,1] = 1
    weight[0,2,1,0] = -3 
    weight[0,2,1,1] = 2.5
    
    weight[0,3,0,0] = 1.5 
    weight[0,3,0,1] = 0.5
    weight[0,3,1,0] = -2 
    weight[0,3,1,1] = 1.5
    
    
    weight[1,0,0,0] = -0.5
    weight[1,0,0,1] = 2.5
    weight[1,0,1,0] = 2.5 
    weight[1,0,1,1] = 0.5
    
    weight[1,1,0,0] = -1.5 
    weight[1,1,0,1] = -1
    weight[1,1,1,0] = 3 
    weight[1,1,1,1] = 0.5
    
    weight[1,2,0,0] = 1.5 
    weight[1,2,0,1] = 1
    weight[1,2,1,0] = 0 
    weight[1,2,1,1] = 0
    
    weight[1,3,0,0] = 1.5 
    weight[1,3,0,1] = 0
    weight[1,3,1,0] = -2 
    weight[1,3,1,1] = 3

    
    weight[2,0,0,0] = 1 
    weight[2,0,0,1] = 1
    weight[2,0,1,0] = 2 
    weight[2,0,1,1] = -2
    
    weight[2,1,0,0] = -1 
    weight[2,1,0,1] = 2
    weight[2,1,1,0] = -3 
    weight[2,1,1,1] = 1
    
    weight[2,2,0,0] = 0 
    weight[2,2,0,1] = 1
    weight[2,2,1,0] = -3 
    weight[2,2,1,1] = 2.5
    
    weight[2,3,0,0] = 1.5 
    weight[2,3,0,1] = 2.5
    weight[2,3,1,0] = -2 
    weight[2,3,1,1] = 1.5
    
    bias = np.ndarray(2)
    
    bias[0] = 0.5
    bias[1] = 1.5
    
    w = [weight, bias]
    model.set_weights(w)
    
    inp = np.ndarray((1,4,5,2))
    
    inp[0,0,0,0] = 0 
    inp[0,0,0,1] = 1
    inp[0,0,1,0] = 2 
    inp[0,0,1,1] = 1
    inp[0,0,2,0] = 0 
    inp[0,0,2,1] = 0
    inp[0,0,3,0] = 2 
    inp[0,0,3,1] = 1
    inp[0,0,4,0] = 2 
    inp[0,0,4,1] = 1
    
    inp[0,1,0,0] = 0 
    inp[0,1,0,1] = -1
    inp[0,1,1,0] = 1 
    inp[0,1,1,1] = -2
    inp[0,1,2,0] = 3 
    inp[0,1,2,1] = 1
    inp[0,1,3,0] = 2 
    inp[0,1,3,1] = 0
    inp[0,1,4,0] = 2 
    inp[0,1,4,1] = -3
    
    inp[0,2,0,0] = 1 
    inp[0,2,0,1] = 2
    inp[0,2,1,0] = -2 
    inp[0,2,1,1] = 0
    inp[0,2,2,0] = 3 
    inp[0,2,2,1] = -3
    inp[0,2,3,0] = 2 
    inp[0,2,3,1] = 1
    inp[0,2,4,0] = 2 
    inp[0,2,4,1] = 0
    
    inp[0,3,0,0] = 1 
    inp[0,3,0,1] = 2
    inp[0,3,1,0] = 0 
    inp[0,3,1,1] = -2
    inp[0,3,2,0] = 3 
    inp[0,3,2,1] = 1
    inp[0,3,3,0] = 2 
    inp[0,3,3,1] = 3
    inp[0,3,4,0] = -3 
    inp[0,3,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_conv_2D_1_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_conv_2D_1_output.json", output.tolist())

def gen_conv_2D_stride_1_2():
    model = Sequential()
    
    model.add(Conv2D(2, (2,4), strides=(2,1), input_shape=(4,5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    weight = np.ndarray((2,4,2,2))
    
    weight[0,0,0,0] = 0 
    weight[0,0,0,1] = 1.5
    weight[0,0,1,0] = 2 
    weight[0,0,1,1] = 0.5
    
    weight[0,1,0,0] = -1 
    weight[0,1,0,1] = -2
    weight[0,1,1,0] = 3 
    weight[0,1,1,1] = 0
    
    weight[0,2,0,0] = 1 
    weight[0,2,0,1] = 1
    weight[0,2,1,0] = -3 
    weight[0,2,1,1] = 2.5
    
    weight[0,3,0,0] = 1.5 
    weight[0,3,0,1] = 0.5
    weight[0,3,1,0] = -2 
    weight[0,3,1,1] = 1.5
    
    
    weight[1,0,0,0] = -0.5
    weight[1,0,0,1] = 2.5
    weight[1,0,1,0] = 2.5 
    weight[1,0,1,1] = 0.5
    
    weight[1,1,0,0] = -1.5 
    weight[1,1,0,1] = -1
    weight[1,1,1,0] = 3 
    weight[1,1,1,1] = 0.5
    
    weight[1,2,0,0] = 1.5 
    weight[1,2,0,1] = 1
    weight[1,2,1,0] = 0 
    weight[1,2,1,1] = 0
    
    weight[1,3,0,0] = 1.5 
    weight[1,3,0,1] = 0
    weight[1,3,1,0] = -2 
    weight[1,3,1,1] = 3
    
    bias = np.ndarray(2)
    
    bias[0] = 0.5
    bias[1] = 1.5
    
    w = [weight, bias]
    model.set_weights(w)
    
    inp = np.ndarray((1,4,5,2))
    
    inp[0,0,0,0] = 0 
    inp[0,0,0,1] = 1
    inp[0,0,1,0] = 2 
    inp[0,0,1,1] = 1
    inp[0,0,2,0] = 0 
    inp[0,0,2,1] = 0
    inp[0,0,3,0] = 2 
    inp[0,0,3,1] = 1
    inp[0,0,4,0] = 2 
    inp[0,0,4,1] = 1
    
    inp[0,1,0,0] = 0 
    inp[0,1,0,1] = -1
    inp[0,1,1,0] = 1 
    inp[0,1,1,1] = -2
    inp[0,1,2,0] = 3 
    inp[0,1,2,1] = 1
    inp[0,1,3,0] = 2 
    inp[0,1,3,1] = 0
    inp[0,1,4,0] = 2 
    inp[0,1,4,1] = -3
    
    inp[0,2,0,0] = 1 
    inp[0,2,0,1] = 2
    inp[0,2,1,0] = -2 
    inp[0,2,1,1] = 0
    inp[0,2,2,0] = 3 
    inp[0,2,2,1] = -3
    inp[0,2,3,0] = 2 
    inp[0,2,3,1] = 1
    inp[0,2,4,0] = 2 
    inp[0,2,4,1] = 0
    
    inp[0,3,0,0] = 1 
    inp[0,3,0,1] = 2
    inp[0,3,1,0] = 0 
    inp[0,3,1,1] = -2
    inp[0,3,2,0] = 3 
    inp[0,3,2,1] = 1
    inp[0,3,3,0] = 2 
    inp[0,3,3,1] = 3
    inp[0,3,4,0] = -3 
    inp[0,3,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_conv_2D_2_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_conv_2D_2_output.json", output.tolist())
    
# MAXPOOLING

# avgpooling1D_tests
def gen_avgpool_1D_stride_1():
    model = Sequential()
    
    model.add(AveragePooling1D(pool_size=3, strides=1, input_shape=(5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,5,2))
    
    inp[0,0,0] = 0 
    inp[0,0,1] = 1
    inp[0,1,0] = 2 
    inp[0,1,1] = 1
    inp[0,2,0] = 0 
    inp[0,2,1] = 0
    inp[0,3,0] = 2 
    inp[0,3,1] = 1
    inp[0,4,0] = 2 
    inp[0,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_avgpool_1D_1_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_avgpool_1D_1_output.json", output.tolist())
    
def gen_avgpool_1D_stride_2():
    model = Sequential()
    
    model.add(AveragePooling1D(pool_size=3, strides=2, input_shape=(5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,5,2))
    
    inp[0,0,0] = 0 
    inp[0,0,1] = 1
    inp[0,1,0] = 2 
    inp[0,1,1] = 1
    inp[0,2,0] = 0 
    inp[0,2,1] = 0
    inp[0,3,0] = 2 
    inp[0,3,1] = 1
    inp[0,4,0] = 2 
    inp[0,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_avgpool_1D_2_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_avgpool_1D_2_output.json", output.tolist())

# avgpooling2D_tests
def gen_avgpool_2D_stride_1_1():
    model = Sequential()
    
    model.add(AveragePooling2D(pool_size=(3,4), strides=(1,1), input_shape=(4,5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,4,5,2))
    
    inp[0,0,0,0] = 0 
    inp[0,0,0,1] = 1
    inp[0,0,1,0] = 2 
    inp[0,0,1,1] = 1
    inp[0,0,2,0] = 0 
    inp[0,0,2,1] = 0
    inp[0,0,3,0] = 2 
    inp[0,0,3,1] = 1
    inp[0,0,4,0] = 2 
    inp[0,0,4,1] = 1
    
    inp[0,1,0,0] = 0 
    inp[0,1,0,1] = -1
    inp[0,1,1,0] = 1 
    inp[0,1,1,1] = -2
    inp[0,1,2,0] = 3 
    inp[0,1,2,1] = 1
    inp[0,1,3,0] = 2 
    inp[0,1,3,1] = 0
    inp[0,1,4,0] = 2 
    inp[0,1,4,1] = -3
    
    inp[0,2,0,0] = 1 
    inp[0,2,0,1] = 2
    inp[0,2,1,0] = -2 
    inp[0,2,1,1] = 0
    inp[0,2,2,0] = 3 
    inp[0,2,2,1] = -3
    inp[0,2,3,0] = 2 
    inp[0,2,3,1] = 1
    inp[0,2,4,0] = 2 
    inp[0,2,4,1] = 0
    
    inp[0,3,0,0] = 1 
    inp[0,3,0,1] = 2
    inp[0,3,1,0] = 0 
    inp[0,3,1,1] = -2
    inp[0,3,2,0] = 3 
    inp[0,3,2,1] = 1
    inp[0,3,3,0] = 2 
    inp[0,3,3,1] = 3
    inp[0,3,4,0] = -3 
    inp[0,3,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_avgpool_2D_1_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_avgpool_2D_1_output.json", output.tolist())
    
def gen_avgpool_2D_stride_1_2():
    model = Sequential()
    
    model.add(AveragePooling2D(pool_size=(3,4), strides=(1,1), input_shape=(4,5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,4,5,2))
    
    inp[0,0,0,0] = 0 
    inp[0,0,0,1] = 1
    inp[0,0,1,0] = 2 
    inp[0,0,1,1] = 1
    inp[0,0,2,0] = 0 
    inp[0,0,2,1] = 0
    inp[0,0,3,0] = 2 
    inp[0,0,3,1] = 1
    inp[0,0,4,0] = 2 
    inp[0,0,4,1] = 1
    
    inp[0,1,0,0] = 0 
    inp[0,1,0,1] = -1
    inp[0,1,1,0] = 1 
    inp[0,1,1,1] = -2
    inp[0,1,2,0] = 3 
    inp[0,1,2,1] = 1
    inp[0,1,3,0] = 2 
    inp[0,1,3,1] = 0
    inp[0,1,4,0] = 2 
    inp[0,1,4,1] = -3
    
    inp[0,2,0,0] = 1 
    inp[0,2,0,1] = 2
    inp[0,2,1,0] = -2 
    inp[0,2,1,1] = 0
    inp[0,2,2,0] = 3 
    inp[0,2,2,1] = -3
    inp[0,2,3,0] = 2 
    inp[0,2,3,1] = 1
    inp[0,2,4,0] = 2 
    inp[0,2,4,1] = 0
    
    inp[0,3,0,0] = 1 
    inp[0,3,0,1] = 2
    inp[0,3,1,0] = 0 
    inp[0,3,1,1] = -2
    inp[0,3,2,0] = 3 
    inp[0,3,2,1] = 1
    inp[0,3,3,0] = 2 
    inp[0,3,3,1] = 3
    inp[0,3,4,0] = -3 
    inp[0,3,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_avgpool_2D_2_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_avgpool_2D_2_output.json", output.tolist())

# maxpooling1D tests
def gen_maxpool_1D_stride_1():
    model = Sequential()
    
    model.add(MaxPooling1D(pool_size=3, strides=1, input_shape=(5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,5,2))
    
    inp[0,0,0] = 0 
    inp[0,0,1] = 1
    inp[0,1,0] = 2 
    inp[0,1,1] = 1
    inp[0,2,0] = 0 
    inp[0,2,1] = 0
    inp[0,3,0] = 2 
    inp[0,3,1] = 1
    inp[0,4,0] = 2 
    inp[0,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_maxpool_1D_1_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_maxpool_1D_1_output.json", output.tolist())
    
def gen_maxpool_1D_stride_2():
    model = Sequential()
    
    model.add(MaxPooling1D(pool_size=3, strides=2, input_shape=(5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,5,2))
    
    inp[0,0,0] = 0 
    inp[0,0,1] = 1
    inp[0,1,0] = 2 
    inp[0,1,1] = 1
    inp[0,2,0] = 0 
    inp[0,2,1] = 0
    inp[0,3,0] = 2 
    inp[0,3,1] = 1
    inp[0,4,0] = 2 
    inp[0,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_maxpool_1D_2_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_maxpool_1D_2_output.json", output.tolist())


def gen_maxpool_2D_stride_1_1():
    model = Sequential()
    
    model.add(MaxPooling2D(pool_size=(3,4), strides=(1,1), input_shape=(4,5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,4,5,2))
    
    inp[0,0,0,0] = 0 
    inp[0,0,0,1] = 1
    inp[0,0,1,0] = 2 
    inp[0,0,1,1] = 1
    inp[0,0,2,0] = 0 
    inp[0,0,2,1] = 0
    inp[0,0,3,0] = 2 
    inp[0,0,3,1] = 1
    inp[0,0,4,0] = 2 
    inp[0,0,4,1] = 1
    
    inp[0,1,0,0] = 0 
    inp[0,1,0,1] = -1
    inp[0,1,1,0] = 1 
    inp[0,1,1,1] = -2
    inp[0,1,2,0] = 3 
    inp[0,1,2,1] = 1
    inp[0,1,3,0] = 2 
    inp[0,1,3,1] = 0
    inp[0,1,4,0] = 2 
    inp[0,1,4,1] = -3
    
    inp[0,2,0,0] = 1 
    inp[0,2,0,1] = 2
    inp[0,2,1,0] = -2 
    inp[0,2,1,1] = 0
    inp[0,2,2,0] = 3 
    inp[0,2,2,1] = -3
    inp[0,2,3,0] = 2 
    inp[0,2,3,1] = 1
    inp[0,2,4,0] = 2 
    inp[0,2,4,1] = 0
    
    inp[0,3,0,0] = 1 
    inp[0,3,0,1] = 2
    inp[0,3,1,0] = 0 
    inp[0,3,1,1] = -2
    inp[0,3,2,0] = 3 
    inp[0,3,2,1] = 1
    inp[0,3,3,0] = 2 
    inp[0,3,3,1] = 3
    inp[0,3,4,0] = -3 
    inp[0,3,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_maxpool_2D_1_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_maxpool_2D_1_output.json", output.tolist())
    
def gen_maxpool_2D_stride_1_2():
    model = Sequential()
    
    model.add(MaxPooling2D(pool_size=(2,4), strides=(2,1), input_shape=(4,5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,4,5,2))
    
    inp[0,0,0,0] = 0 
    inp[0,0,0,1] = 1
    inp[0,0,1,0] = 2 
    inp[0,0,1,1] = 1
    inp[0,0,2,0] = 0 
    inp[0,0,2,1] = 0
    inp[0,0,3,0] = 2 
    inp[0,0,3,1] = 1
    inp[0,0,4,0] = 2 
    inp[0,0,4,1] = 1
    
    inp[0,1,0,0] = 0 
    inp[0,1,0,1] = -1
    inp[0,1,1,0] = 1 
    inp[0,1,1,1] = -2
    inp[0,1,2,0] = 3 
    inp[0,1,2,1] = 1
    inp[0,1,3,0] = 2 
    inp[0,1,3,1] = 0
    inp[0,1,4,0] = 2 
    inp[0,1,4,1] = -3
    
    inp[0,2,0,0] = 1 
    inp[0,2,0,1] = 2
    inp[0,2,1,0] = -2 
    inp[0,2,1,1] = 0
    inp[0,2,2,0] = 3 
    inp[0,2,2,1] = -3
    inp[0,2,3,0] = 2 
    inp[0,2,3,1] = 1
    inp[0,2,4,0] = 2 
    inp[0,2,4,1] = 0
    
    inp[0,3,0,0] = 1 
    inp[0,3,0,1] = 2
    inp[0,3,1,0] = 0 
    inp[0,3,1,1] = -2
    inp[0,3,2,0] = 3 
    inp[0,3,2,1] = 1
    inp[0,3,3,0] = 2 
    inp[0,3,3,1] = 3
    inp[0,3,4,0] = -3 
    inp[0,3,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_maxpool_2D_2_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_maxpool_2D_2_output.json", output.tolist())    


# FLATTEN
def gen_flatten():
    model = Sequential()
    
    model.add(Flatten(input_shape=(4,5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,4,5,2))
    
    inp[0,0,0,0] = 0 
    inp[0,0,0,1] = 1
    inp[0,0,1,0] = 2 
    inp[0,0,1,1] = 1
    inp[0,0,2,0] = 0 
    inp[0,0,2,1] = 0
    inp[0,0,3,0] = 2 
    inp[0,0,3,1] = 1
    inp[0,0,4,0] = 2 
    inp[0,0,4,1] = 1
    
    inp[0,1,0,0] = 0 
    inp[0,1,0,1] = -1
    inp[0,1,1,0] = 1 
    inp[0,1,1,1] = -2
    inp[0,1,2,0] = 3 
    inp[0,1,2,1] = 1
    inp[0,1,3,0] = 2 
    inp[0,1,3,1] = 0
    inp[0,1,4,0] = 2 
    inp[0,1,4,1] = -3
    
    inp[0,2,0,0] = 1 
    inp[0,2,0,1] = 2
    inp[0,2,1,0] = -2 
    inp[0,2,1,1] = 0
    inp[0,2,2,0] = 3 
    inp[0,2,2,1] = -3
    inp[0,2,3,0] = 2 
    inp[0,2,3,1] = 1
    inp[0,2,4,0] = 2 
    inp[0,2,4,1] = 0
    
    inp[0,3,0,0] = 1 
    inp[0,3,0,1] = 2
    inp[0,3,1,0] = 0 
    inp[0,3,1,1] = -2
    inp[0,3,2,0] = 3 
    inp[0,3,2,1] = 1
    inp[0,3,3,0] = 2 
    inp[0,3,3,1] = 3
    inp[0,3,4,0] = -3 
    inp[0,3,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_flat_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_flat_output.json", output.tolist())

# DENSE
def gen_dense_units_4():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
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
    
    write("tests/test_dense_output.json", output.tolist())

# -------------------------------------------------------------------------------------------------------------

# ACTIVATIONS:

# ELu test
def gen_elu():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('elu'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
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
    
    write("tests/test_elu_output.json", output.tolist())

# HardSigmoid test
def gen_hard_sigmoid():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('hard_sigmoid'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
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
    
    write("tests/test_hard_sigmoid_output.json", output.tolist())

# ReLu test
def gen_relu():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('relu'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
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
    
    write("tests/test_relu_output.json", output.tolist())

# Sigmoid test
def gen_sigmoid():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
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
    
    write("tests/test_sigmoid_output.json", output.tolist())

# Softmax test
def gen_softmax():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
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
    
    write("tests/test_softmax_output.json", output.tolist())

# SoftPlus test
def gen_softplus():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('softplus'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
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
    
    write("tests/test_softplus_output.json", output.tolist())

# SoftSign test
def gen_softsign():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('softsign'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
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
    
    write("tests/test_softsign_output.json", output.tolist())

# TanH test
def gen_tanh():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('tanh'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
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
    
    write("tests/test_tanh_output.json", output.tolist())


# Generate ALL the tests:

generate_test_files()
