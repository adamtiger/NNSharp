The main idea behind a recurrent neural network (RNN) is to grab the relations in a process where the input is a time-sequence with consequtive data pieces. Therefore the order of the pieces is strict. The relation connects to the time. An examplary problem can be when a software tries to predict the next data piece by knowing the previous 5 pieces. 

Generally the input of an rnn is a set of vectors (e.g.: a vector for each time point) and the output is also a set of vector but may be it has different size then the input. Each vector is processed by a cell (only one type of cell is applied) but in a strict order and with some delay. Therefore as the computation goes each cell processes a vector, gives (or not) an output and calculates some further state values which will be fed into the next cell. The next cell receives the state values and the next input vector from the sequence, executes the calculations and so on. 

Shortly, an RNN is a finite long sequence of cells. Each cell has the same structure. The input vectors processed consequtively and some hidden variables are computed and forwarded as well. Some references for further insight:

[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)<br>
[A Beginner’s Guide to Recurrent Networks and LSTMs](https://deeplearning4j.org/lstm#a-beginners-guide-to-recurrent-networks-and-lstms)<br>
[Recurrent Neural Networks Tutorial on WILDML](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/).


<span style="float:right;">[[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/SimpleRNNKernel.cs#L12)</span>
## SimpleRNN

The structure of the so called simple RNN layer in Keras is the following:

![simplernn](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwUXZjSUVOUjBJZEE)

The **h** is the hidden variable, **R** is the recurrent kernel, **W** is the kernel, **b** is the bias and **f** is the activation function. 

**Input:**

A Data2D type with the shape: (1, timesteps, input dimension, batches).

**Output:**

A Data2D type with shape: (1, 1, units, batches). 

**Methods:**

The SimpleRNN cell implements the operations shown in the picture and feeds the output to the next cell. The unrolled cells are implemented by a *for* cycle. It can be useful to inspect the [Keras implementation](https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L555) as well.

<span style="float:right;">[[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/LSTMKernel.cs#L11)</span>
## LSTM

The structure of the LSTM cell is the following:

![lstm](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwMFZWRFJrSXFtd28)

The **g** is the recurrent activation, **p** is the activation, **W**s are the kernels, **U**s are the recurrent kernels, **h** is the hidden variable which is the output too and the notation * is an element-wise multiplication.

**Input:**

A Data2D type with the shape: (1, timesteps, input dimension, batches).

**Output:**

A Data2D type with shape: (1, 1, units, batches).

**Methods:**

The Keras implementation can help as well: [see](https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L1123) the step function in the LSTM implementation. Basically the products are dot products between matricies.

<span style="float:right;">[[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/GRUKernel.cs#L11)</span>
## GRU

The structure of the GRU cell is the following:

![gru](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwcU4tVVVQS0d3VmM)

The meaning of the notations are the same as in case of LSTM. 1-z means an element-wise subtraction. 

**Input:**

A Data2D type with the shape: (1, timesteps, input dimension, batches).

**Output:**

A Data2D type with shape: (1, 1, units, batches).

**Methods:**

The Keras implementation can help as well: [see](https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L828) the step function in the LSTM implementation. [This](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/) blog article can be useful as well.
