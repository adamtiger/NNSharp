
<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/DataTypes/Data2D.cs) </span>
## Data2D

This is a data type for stroring data with 2 dimensions like images. It has 2 further dimensions for storing the channels and batches. Therefore this type has four indicies with the following meaning: (height, width, channel, batch). The available functions:

	void ApplyToAll(Operation op)

Operation is a function which receives a *double* and gives back a *double*. The operation will be applied on each element in the data independently.

	void ToZeros()

All of the elements become zero after applying this function.

	Dimension GetDimension()

The Dimension is a structure with the following public attributes *h*, *w*, *c*, *b*.

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/DataTypes/DataArray.cs) </span>
## DataArray

This is a special array. The data is accessable by one index. It has the following functions:

	void ApplyToAll(Operation op)

The same as before.

	void ToZeros()

The same as before.

	int GetLength()

The length of the array.

This array is enumerable as well.

<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/master/NNSharp/DataTypes/SequentialModelData.cs) </span>
## SequentialModelData

equentialModelData has the following methods to access information about the layers and the model:

	int GetNumberofLayers()

Gives the number of layers in the model. The InputLayer is not taken into account.

	string GetLayerNameAt(int idx)

Gives the name of the layer at place *idx*. The counting starts at zero.

	double GetExecutionTime()

Gives the evaluation time of the network for one forward pass in *milli seconds*.

	string GetStringRepresentation()

Gives a string containing all the information of the layers.

	LayerData GetLayerDataAt(int idx)

Gives the LayerData for a concrete layer. LayerData contains the parameters of the input and the output data. The name of the layer is also available.


