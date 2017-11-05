// TryCNTKcpp.cpp : Defines the entry point for the console application.
//

#include <vector>
#include <iostream>

//region: TF MINI TEST

#define COMPILER_MSVC
#include "tensorflow.h"
#include "nnsharp.h"

void tf_test() {
	
	std::cout << "Hi!" << std::endl;

	const char* version = TF_Version();
	std::cout << version[0] << version[1] << version[2] << version[3] << version[4] << std::endl;
	TF_Buffer* ops = TF_GetAllOpList();
	int d;
	std::cin >> d;
	
}

// endregion (TF MINI TEST)

int main(int argc, char *argv[])
{

	//CNTK_test();
	//tf_test();
	

    return 0;
}

// DLL test

core::tensor::Tensor::Tensor(DataType data_type, int dim, int* shape) {

	if (data_type == DataType::Integer) {

		// Calculate the size.
		int length = 1;
		for (int i = 0; i < dim; ++i) {
			length *= shape[i];
		}

		values = new int[length];
		this->dim = dim;
		this->shape = shape;
	}
	else
		values = nullptr;
}

core::tensor::Tensor::~Tensor() {
	if (values != nullptr)
		delete[] values;
}

core::tensor::TensorInteger::TensorInteger(int size):
	Tensor(DataType::Integer, 1, &size){
}

void core::tensor::TensorInteger::Get(int indices, int* value_out) const {
	const char* version = TF_Version();
	std::cout << version[0] << version[1] << version[2] << version[3] << version[4] << std::endl;
	*value_out = version[1];
}

void core::tensor::TensorInteger::Set(int idx, int value_in){

}

core::tensor::TensorInteger* core::tensor::create_tensor_integer(int size)
{
	return new TensorInteger(size);
}

int core::tensor::tensor_integer_get(TensorInteger * tensor_in, int idx)
{
	int retVal = 0;
	tensor_in->Get(idx, &retVal);

	return retVal;
}



