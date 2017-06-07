Activation functions are applied in an element-wise maner on the input. *Therefore the dimension of the input and output is the same.* The goal of the activation functions is to break the linearity of the network. This can provide a function space (in terms of the trainable weights) which can approximate a large body of mapping rules from the inputs to the outputs. For mathematical details see the following references: 

* Funahashi, K. I. "On the Approximate Realization of Continuous Mappings by Neural Networks", Neural Networks, Vol. 2. No. 3. pp. 183-192. 1989.
* Leshno, M. - Lin, V. Y. - Pinkus, A. - Schocken, S. "Multilayer Feedforward Networks With a Nonpolynomial Activation Function Can Approximate Any Function", Neural Networks, Vol. 6. pp. 861-867. 1993.

The last one contains the following **theorem**:

Assume the neural network contains one non-linear layer. An *f*  function (domain and range are real numbers) can be approximated with arbitrary accuracy if and only if the applied activation function is not a polynom and it is locally bounded.

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/ELuKernel.cs) </span>
## ELu

![elugraph](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwUjZKVldXSHAtc0U)

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/HardSigmoidKernel.cs) </span>
## HardSigmoid

![hardsigmoidgraph](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwNFo2YloxbEFhT00)

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/ReLuKernel.cs) </span>
## ReLu

![relugraph](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwRFlVSnpoUXM4X2c)

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/SigmoidKernel.cs) </span>
## Sigmoid

![sigmoidgraph](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwc2FmSmhiTEt3N0E)

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/SoftPlusKernel.cs) </span>
## SoftPlus

![softplusgraph](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwQXNOaXBRZGxSUjA)

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/SoftmaxKernel.cs) </span>
## Softmax

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/SoftsignKernel.cs) </span>
## Softsign

![softsigngraph](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwTUppTHhKeEVKTkk)

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/TanHKernel.cs) </span>
## TanH

![tanhgraph](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwLU1zQlBqbldaX3c)
