# Highlevel C++ API for deep learning

## Vision

The goal of this subproject is to create a highlevel C++ API which wraps the Tensorflow and CNTK C++ APIs in a Keras like manner and with some extensions. This can be two major advantages:

* it is easier to bind it to other languages and get a high level API in them without further implementations
* performance losses can be avoided due to the C++ implementation behind the API.

## Platform independency

This subproject can be treated independently from NNSharp which is a C# oriented deep learning library. The C++ API is a cmake-based project and provides platform independent builds. Therefore the API can be used on any platform.

The necessary third party shared libraries can be downloaded from the following links:

[CNTK-Windows](https://drive.google.com/file/d/0B97L9zqg-lnwci1CSVFyRDd5ZWM/view?usp=sharing)

[CNTK-Linux](https://drive.google.com/open?id=0B97L9zqg-lnwUTM4ZC16QlVUZUk) 

[Tensorflow-Windows](https://drive.google.com/open?id=0B97L9zqg-lnwejFEeVRWY1MtUGs)

[Tensorflow-Linux](https://drive.google.com/open?id=0B97L9zqg-lnwNEE2UDZIUXFhX1U)