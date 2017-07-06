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

Gives an object of **SequentialModelData** type. Therefore it is necessary to cast the output for that type. For the details of **SequentialModelData** see the documentation's *Data structure* part.


## Graph model
