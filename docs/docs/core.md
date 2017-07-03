The core layers contain layers to transform the shape of the input and the fully connected layer as a usual ingridient of a neural network. 

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/Dense2DKernel.cs) </span>
## Fully connected (Dense layer) 

The structure of *Fully connected layer* can be seen on the following image:

![dense](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwdUtReFNsTHl4Rlk)

**Input:**

A Data2D type with the shape: (1, 1, channels, batches).

**Output:**

A Data2D type with the shape: (1, 1, units, batches).

**Methods:**

As it can be seen on the picture this layer uses as many weight vectors (kernels) as many units have the layer. The number of units is equal with the number of output units. The fully connected layer is a linear transformation. A weight multiplies the corresponding input value (blue circle), then the output value (orange circle) is the sum of the previously weighted inputs. For further details see the source code where kernels are used in terms of units.

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/FlattenKernel.cs) </span>
## Flatten

The *Flatten layer* creates a 1-dimensional output (array) from the input. 

**Input:**

A Data2D type data with arbitrary shape (rows, columns, channels, batches can be anything).

**Output:**

A Data2D type data with the following shape: (1, 1, rows * columns * channels, batches). This represents a 1-dimensional vector.

**Method:**

The mapping occurs in the following way. The `batch` remains unchanged. For a given `batch` the reading order is: 1) channel, 2) column, 3) row. Therefore when `row` and `column` are fixed, `channel` is changing. When `channel` achieves its bound, `column` is increased while `row` is still fixed. Then `channel` is iterated again and so on.

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/Reshape2DKernel.cs) </span>
## Reshape2D

The *Reshape2D layer* creates an output with the prescribed shape.

**Input:**

A Data2D type data with arbitrary shape (rows, columns, channels, batches can be anything).

**Output:**

A Data2D type data with the required (prescribed) shape.

**Method:**

Reshaping assumes a strict reading order for accessing all of the elements. It has the following priority regarding the dimensions: 1) channel, 2) column, 3) row, 4) batch. If the elements are accessed by this manner, a one dimensional array (let's call it *virtual array*) will contain all the elements. The algorithm will reshape the original data into a new one that after using the same access method on it, the same virtual array would appear. (The same element in the original and the new data should be mapped to the same place in the virtual array by the access method.)

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/PermuteKernel.cs) </span>
## Permute

The *Permute layer* creates an output with the same number of elements but the roles of the dimensions are changed.

**Input:**

A Data2D type data with arbitrary shape (rows, columns, channels, batches can be anything).

**Output:**

A Data2D type data with permuted shape (new rows, new columns, new channels, batches). The *batches can not* be changed by permutation. 

**Method:**

Let's suppose that the rows and channels are permuted in the output. Then an element in the input will be written to the permuted places as the example shows:

	Indicies in the input: h, w, c, b --> element = input[h, w, c, b].
	Then in the output data: output[c, w, h, b] = element.


<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/RepeatVectorKernel.cs) </span>
## RepeatVector

The *RepeatVector layer* repeats a 1-dimensional vector n times.

**Input:**

A Data2D type data with shape: (1, 1, channels, batches).

**Output:**

A Data2D type data with shape: (1, n, channels, batches).

**Method:**

Repeats the same 1-dimensional input for each channel and batch. 

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/Bias2DKernel.cs#L12) </span>
## Bias

The *Bias layer* adds bias values to the input data. Elements with the same channel are increased by the same bias value. The input and the output shapes *are the same*.

**Input:**

A Data2D type data with arbitrary shape (rows, columns, channels, batches can be anything).

**Output:**

It is the same as the input.

**Method:**

Adding bias  means the following:

	output[h, w, c, b] += input[h, w, c, b] + bias[c]

