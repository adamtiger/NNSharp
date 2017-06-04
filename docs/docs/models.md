# Models

## Interface: IModel

This is the interface for all of the models. It contains only one function which returns the structure of the model. This is important to get access to the model details.

## Sequential model
<span style="float:right;"> [[source]](https://github.com/adamtiger/NNSharp/blob/dev/NNSharp/Models/SequentialModel.cs#L15) </span>

<br>
The smallest building block of the sequential model is the **layer**. The layers are organised consequtively. The output of a layer is the input to the next layer.

NNSharp architecture is based on general descriptors of the possible operations (called kernels in this context) and then it applies a compiler to get the executable model. The compilation is done when a reader function (see Keras part for example) is called. The reader function reads the descriptors, builds a model then compiles it. After compilation calculations on new date reuires to call:

	IData ExecuteNetwork(IData input)

Where IData can be a Data2D data type (or later Data3D). At the detailed descriptions the exact meaning of the data format for the particular operation is described. 

	Dimension GetInputDimension()

Gives a structure, Dimension, containing the dimension of the expected input data.

	IModelData GetSummary()

Gives an object of **SequentialModelData** type. Therefore it is necessary to cast the output for that type. SequentialModelData has the following methods to access information about the layers and the model:

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


## Graph model
