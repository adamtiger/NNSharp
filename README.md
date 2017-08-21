![logo](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnweGpHSXFoT2JWS2c)

[![Build status](https://ci.appveyor.com/api/projects/status/m7albu3gen3orswj/branch/master?svg=true)](https://ci.appveyor.com/project/adamtiger/nnsharp/branch/master)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/adamtiger/NNSharp/blob/master/LICENSE)

A lightweight package for running pre-trained neural networks. For the **detailed documentation** see the [NNSharp documentation](https://adamtiger.github.io/NNSharp/). 

## Philosophy

This library was created in order to run pre-trained neural networks. Training is the most time consuming part of a deep learning framework. Probably C# is not the best for training neural networks. There are a lot of very good solution can be found like [Tensorflow](https://www.tensorflow.org/), [CNTK](https://www.cntk.ai/pythondocs/index.html), [Theano](http://deeplearning.net/software/theano/), [PyTorch](http://pytorch.org/), [Sonnet](https://github.com/deepmind/sonnet) ans so on. Most of them suit the Python programming language and mainly support Linux. 

Therefore this library aims for using the weights and structure created with the above softwares and just run it in C# on **Windows**.

## Installation

The package is [available](https://www.nuget.org/packages/NNSharp/) as a NuGet package from nuget.org. The current NuGet package was built on Windows8.1, Visual Studio 2015 and .Net Framework4.5.2.

## Getting started

It is very easy to use the library. The master branch contains a python script named KerasModeltoJSON.py. It takes the created and compiled Keras *model* and the *output file name* as arguments. The json file can be created as follows:

```python
import KerasModeltoJSON as js
wrt = js.JSONwriter(model, fname)
wrt.save()
```

Then in the C# program:

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

For the detailed documentation see the [NNSharp documentation](https://adamtiger.github.io/NNSharp/). 

## Plans and contributions

The roadmap of the development can be seen on the following picture:

![roadmap](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwZnRPQVFhYzZDMDQ  "Roadmap")

The first stage is almost finished. The C++ API is planned to be similar like the Keras API in Python but with some further extensions (concrete architectures, file readers from different library outputs, wrapper for threading etc.)

For the current tasks see the issues. If you want to contribute write a letter to the following address: adam.8.budai at gmail dot com. 

## Summary

This library aims for connecting the models trained in Python with Tensorflow, Theano, Pytorch, Sonnet with C#. It is able to run the models but training has no priority in this project.

## Licence

This project runs under the [MIT licence](https://github.com/adamtiger/NNSharp/blob/master/LICENSE). You can use it in any Open Source project.

