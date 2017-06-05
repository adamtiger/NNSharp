# Getting Started with Keras models

If the training was done with Keras then the model can be saved by using the python script named **KerasModeltoJSON.py**. It will create a json file which can be read by NNSharp. NNSharp will build the model in C# and provides function to execute it. The python script takes the created and compiled Keras *model* and the *output file name* as arguments. Then the json file can be created in the following way in your Python program:

```python
import KerasModeltoJSON as js
wrt = js.JSONwriter(model, file_path)
wrt.save()
```

Then in the C# program use the following:

```csharp
// Read the previously created json.
var reader = new ReaderKerasModel(filePath); 
SequentialModel model = reader.GetSequentialExecutor();

// Then create the data to run the executer on.
// batch: should be set in the Keras model.
Data2D input = new Data2D(height, width, channel, batch);

// Calculate the network's output.
IData output = model.ExecuteNetwork(input);
```

In order to know what data should be the *input* and the *output* see the corresponding documentations for the *layer at input* and the *layer at output*, respectively.
