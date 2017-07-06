
<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/Conv1DKernel.cs) </span>
## Convolution1D 

1-dimensional convolution is a special case of a 2-dimensional convolution. It processes the input with a sliding window (called kernel).

**Input:**

A Data2D type data with shape: (1, length, channels, batches).

**Output:**

A Data2D type data with shape: (1, new length, kernel numbers, batches). 

The shape of the filter: (1, kernel size, channels, kernel numbers). Then the output length (new  length):
`new length = 1 + (length + 2 * padding - kernel size)/stride`.

**Methods:**

The 1-dimensional convolution applies a 1-diemsional kernel with the same number of channels than the input has. Then the kernel moves around the input vector and multiplies the corresponding elements then the results of the multiplications are summed up. The kernel starts at the left side of the input tensor. If **padding** is different than 0, the input tensor is virtually extended by some padding value (usually 0) at the left and right sides. The kernel moves some places ahead to the direction of the right side. The number of places are defined by the **stride**. The formula of the calculation: `output(i) = sum_c(sum_k(input(i * stride - padding + k, c) * kernel(k, c)))`, where i is the place in the output, k is the index of an element, c is the index of the channel. 

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/Conv2DKernel.cs) </span>
## Convolution2D

An illustration of the calculation for a given position of the kernel.

![conv2d](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwZWVNUkV5bDV3YXc)

**Input:**

A Data2D type data with shape: (height, width, channels, batches).

**Output:**

A Data2D type data with shape: (new height, new width, kernel numbers, batches). 

The shape of the filter: (kernel height, kernel width, channels, kernel numbers). Then the output sizes:
`new height = 1 + (height + 2 * padding vertical - kernel height)/stride vertical`,
`new width = 1 + (width + 2 * padding horizontal - kernel width)/stride horizontal`.

**Methods:**

The same as the one dimensional but the sliding happens in the second dimension as well. The formula:
`x = i * strideHZ - paddingHZ + kHZ` <br>
`y = j * strideVR - paddingVR + kVR`
`output(i,j) = sum_c(sum_kHZ(sum_kVR((x, y, c) * kernel(i, j, c))))`

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/Cropping1DKernel.cs) </span>
## Cropping1D 

Trims some elements at the beginning and at the end.

**Input:**

A Data2D type data with shape: (1, length, channels, batches).

**Output:**

A Data2D type data with shape: (1, length - trimmed, channels, batches).

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/Cropping2DKernel.cs) </span>
## Cropping2D

Trims elements at the top, bottom, left and right sides. 

**Input:**

A Data2D type data with shape: (height, width, channels, batches).

**Output:**

A Data2D type data with shape: (height - trimmedA, width - trimmedB, channels, batches).

trimmedA is the overall trimmed rows at the top and bottom. <br>
trimmedB is the overall trimmed columns at the left and right.
