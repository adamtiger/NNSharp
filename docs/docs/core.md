The core layers contain layers to transform the shape of the input and the fully connected layer as a usual ingridient of a neural networks. 

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/Dense2DKernel.cs) </span>
## Fully connected (Dense layer) 

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
(Next release)

The *Reshape2D layer* creates an output with the prescribed shape.

**Input:**

A Data2D type data with arbitrary shape (rows, columns, channels, batches can be anything).

**Output:**

A Data2D type data with the required (prescribed) shape.

**Method:**

Reshaping assumes a strict reading order for accessing all of the elements. It has the following priority regarding the dimensions: 1) channel, 2) column, 3) row, 4) batch. If the elements are accessed by this manner, a one dimensional array (let's call it *virtual array*) will contain all the elements. The algorithm will reshape the original data into a new one that after using the same access method on it, the same virtual array would appear. (The same element in the original and the new data should be mapped to the same place in the virtual array by the access method.)

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/PermuteKernel.cs) </span>
## Permute
(Next release)

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
(Next release)

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/Bias2DKernel.cs#L12) </span>
## Bias


