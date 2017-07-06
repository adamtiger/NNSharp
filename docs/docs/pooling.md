The pooling layers similar to the convolutional layers except an important difference: the pooling occurs for the channels separately.

<span style="float:right;">[[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/AvgPool1DKernel.cs#L12)</span>
## AveragePooling1D

This layer calculates the average value in the kernel (sliding window). 

**Input:**

A Data2D type data with shape: (1, length, channels, batches).

**Output:**

A Data2D type data with shape: (1, new length, channels, batches). 

The shape of the filter: (1, kernel size, 1, 1). Then the output length (new  length):
`new length = 1 + (length + 2 * padding - kernel size)/stride`.


<span style="float:right;">[[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/AvgPool2DKernel.cs#L12)</span>
## AveragePooling2D

**Input:**

A Data2D type data with shape: (height, width, channels, batches).

**Output:**

A Data2D type data with shape: (new height, new width, channels, batches). 

The shape of the filter: (kernel height, kernel width, 1, 1). Then the output sizes:
`new height = 1 + (height + 2 * padding vertical - kernel height)/stride vertical`,
`new width = 1 + (width + 2 * padding horizontal - kernel width)/stride horizontal`.

<span style="float:right;">[[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/GlobalAvgPool1DKernel.cs#L12)</span>
## GlobalAveragePooling1D

Calculates the average value for each channel in the input data. 

**Input:**

A Data2D type data with shape: (1, length, channels, batches).

**Output:**

A Data2D type data with shape: (1, 1, channels, batches). 

<span style="float:right;">[[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/GlobalAvgPool2DKernel.cs#L12)</span>
## GlobalAveragePooling2D

Calculates the average value for each channel in the input data. 

**Input:**

A Data2D type data with shape: (height, width, channels, batches).

**Output:**

A Data2D type data with shape: (1, 1, channels, batches).

<span style="float:right;">[[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/GlobalMaxPool1DKernel.cs#L12)</span>
## GlobalMaxPooling1D

Calculates the maximum value for each channel in the input data. 

**Input:**

A Data2D type data with shape: (1, length, channels, batches).

**Output:**

A Data2D type data with shape: (1, 1, channels, batches). 

<span style="float:right;">[[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/GlobalMaxPool2DKernel.cs#L12)</span>
## GlobalMaxPooling2D

Calculates the maximum value for each channel in the input data. 

**Input:**

A Data2D type data with shape: (height, width, channels, batches).

**Output:**

A Data2D type data with shape: (1, 1, channels, batches).

<span style="float:right;">[[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/MaxPool1DKernel.cs#L12)</span>
## MaxPooling1D

This layer calculates the maximum value in the kernel (sliding window). 

**Input:**

A Data2D type data with shape: (1, length, channels, batches).

**Output:**

A Data2D type data with shape: (1, new length, channels, batches). 

The shape of the filter: (1, kernel size, 1, 1). Then the output length (new  length):
`new length = 1 + (length + 2 * padding - kernel size)/stride`.

<span style="float:right;">[[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/MaxPool2DKernel.cs#L12)</span>
## MaxPooling2D

This layer calculates the maximum value in the kernel (sliding window).

**Input:**

A Data2D type data with shape: (height, width, channels, batches).

**Output:**

A Data2D type data with shape: (new height, new width, channels, batches). 

The shape of the filter: (kernel height, kernel width, 1, 1). Then the output sizes:
`new height = 1 + (height + 2 * padding vertical - kernel height)/stride vertical`,
`new width = 1 + (width + 2 * padding horizontal - kernel width)/stride horizontal`.

