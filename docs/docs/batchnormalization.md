<span style="float:right;">[[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/Kernels/CPUKernels/BatchNormKernel.cs#L11) </span>
## Batch Normalization 

The intuition behind Batch Normalization is that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. This tends to slow down the learning process. Normalization transforms the input data in a way that its mean becomes 0 and its standard deviation becomes 1. For further details see the original article:

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167).

**Input:**

A Data2D array with arbitrary shape.

**Output:**

The same as the input. The numbers are normalized in the channels separately. 

**Methods:**

The following equations are implemented. *The normalization is performed according to the feature (channel) axis.*

![batchnorm](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwaTlqbEZvOXpCOGM)

The **gamma**, **beta** are parameters to learn, **sigma** is the variance, **b** is the bias. The **epsilon** is a small number to avoid the problem with small variances. (Ensures numerical stability.)
