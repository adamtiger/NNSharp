#ifndef __NNSHARP__
#define __NNSHARP__

#if defined (COMPILER_MSVC)
#define CORE_EXPORT __declspec(dllexport)
#else 
#define CORE_EXPORT __attribute__((visibility("default")))
#endif

	// #include <CNTKLibrary.h> These will be included only in the cpp files. 
	// #include <Tesorflow.h>
#include <vector>

	/*
	This is just a sketch for the NNShapCppCore.
	--- The main point is to: ---
		-> create all of the functions
		-> test the build process on ALL the platforms
		-> create the test environment: 1) functional, 2) accuracy and 3) performance
		-> create a tensor that can be effectively binded to C# (test this, important!)
	This file is going to be the bases for the later version which is going to go public on github.

	--- Phylisophy: ---
		-> Keras like API with extension
		-> simple C in order to avoid compiler problems and make easier the binding
		-> fast data structure

	--- Components: ---
		-> Tensor as a data structure
		-> Context to use TF or CNTK (? if the compilation is possible for both of them in the same file)
		-> Status to indicating the success of a calculation (everything goes into the argument list)
		-> functions for forward pass and training work
	*/

	// Tensor

	// A tensor has shape and datatype.
	// Avoid template, instead use a factory function and enumerator to create the right function.
	// Types: int, float, double, binary (? how to represent it effectively ?)
	// 2 and 3 D Tensors separately and a general one

	namespace core {

		struct CORE_EXPORT nnsharp_status {

			nnsharp_status(int id, char* message) :
				id(id), message(message) {}

			int id;
			char* message;
		};
		typedef nnsharp_status* nnsharp_status_p;

		namespace tensor {

			enum class DataType
			{
				Integer,
				Float,
				Double,
				Binary
			};

			struct Tensor {

			public:
				Tensor(DataType data_type, size_t dim, size_t* shape);

				// Creaters.
				//static Tensor* Create(); // Creates a default tensor with 1 dimension and shape {1}. Value is 0.
				/*static Tensor* Create(DataType data_type, size_t dim, size_t* shape); // All values are zeros.
				static Tensor* Ones(DataType data_type, size_t dim, size_t* shape); // All values are ones.
				static Tensor* Random(DataType data_type, size_t dim, size_t* shape); // Creates a tensor with random values (0, 1). For test purposes.
				*/
				~Tensor();

			protected:

				DataType data_type;
				void* values;
				size_t dim;
				size_t* shape;
			};

			class CORE_EXPORT TensorInteger : public Tensor {

			public:
				TensorInteger(size_t size);

				// Access
				void Get(size_t* indices, int* out_value) const;
				void Set(size_t* indices, int in_value);
			};

			CORE_EXPORT extern nnsharp_status_p create_tensor_integer(int size, TensorInteger* tensor_ou);
			/*CORE_EXPORT extern nnsharp_status_p ones_tensor_integer(int size, Tensor* tensor_ou);
			CORE_EXPORT extern nnsharp_status_p random_tensor_integer(int size, Tensor* tensor_ou);
			CORE_EXPORT extern nnsharp_status_p destroy_tensor_integer(Tensor* tensor);*/

			/*class TensorInteger2D : public Tensor {

			public:

				static TensorInteger2D* Create();
				static TensorInteger2D* Create(size_t* shape); // All values are zeros.
				static TensorInteger2D* Ones(size_t* shape); // All values are ones.
				static TensorInteger2D* Random(size_t* shape); // Creates a tensor with random values (0, 1). For test purposes.

															  // Access
				void Get(size_t* indices, int* out_value) const;
				void Set(size_t* indices, int in_value);
			};

			class TensorInteger3D : public Tensor {

			public:

				static TensorInteger3D* Create();
				static TensorInteger3D* Create(size_t* shape); // All values are zeros.
				static TensorInteger3D* Ones(size_t* shape); // All values are ones.
				static TensorInteger3D* Random(size_t* shape); // Creates a tensor with random values (0, 1). For test purposes.

															  // Access
				void Get(size_t* indices, double* out_value) const;
				void Set(size_t* indices, double in_value);
			};


			class TensorDouble : public Tensor {

			public:

				static TensorDouble* Create();
				static TensorDouble* Create(size_t dim, size_t* shape); // All values are zeros.
				static TensorDouble* Ones(size_t dim, size_t* shape); // All values are ones.
				static TensorDouble* Random(size_t dim, size_t* shape); // Creates a tensor with random values (0, 1). For test purposes.

																		 // Access
				void Get(size_t* indices, int* out_value) const;
				void Set(size_t* indices, int in_value);
			};

			class TensorDouble2D : public Tensor {

			public:

				static TensorDouble2D* Create();
				static TensorDouble2D* Create(size_t* shape); // All values are zeros.
				static TensorDouble2D* Ones(size_t* shape); // All values are ones.
				static TensorDouble2D* Random(size_t* shape); // Creates a tensor with random values (0, 1). For test purposes.

				// Access
				void Get(size_t* indices, double* out_value) const;
				void Set(size_t* indices, double in_value);
			};

			class TensorDouble3D : public Tensor {

			public:

				static TensorDouble3D* Create();
				static TensorDouble3D* Create(size_t* shape); // All values are zeros.
				static TensorDouble3D* Ones(size_t* shape); // All values are ones.
				static TensorDouble3D* Random(size_t* shape); // Creates a tensor with random values (0, 1). For test purposes.

													  // Access
				void Get(size_t* indices, double* out_value) const;
				void Set(size_t* indices, double in_value);
			};


			class TensorFloat : public Tensor {

			public:

				static TensorFloat* Create();
				static TensorFloat* Create(size_t dim, size_t* shape); // All values are zeros.
				static TensorFloat* Ones(size_t dim, size_t* shape); // All values are ones.
				static TensorFloat* Random(size_t dim, size_t* shape); // Creates a tensor with random values (0, 1). For test purposes.

																		// Access
				void Get(size_t* indices, int* out_value) const;
				void Set(size_t* indices, int in_value);
			};

			class TensorFloat2D {

			public:

				static TensorFloat2D* Create();
				static TensorFloat2D* Create(size_t* shape); // All values are zeros.
				static TensorFloat2D* Ones(size_t* shape); // All values are ones.
				static TensorFloat2D* Random(size_t* shape); // Creates a tensor with random values (0, 1). For test purposes.

															  // Access
				void Get(size_t* indices, float* out_value) const;
				void Set(size_t* indices, float in_value);
			};

			class TensorFloat3D {

			public:

				static TensorFloat3D* Create();
				static TensorFloat3D* Create(size_t* shape); // All values are zeros.
				static TensorFloat3D* Ones(size_t* shape); // All values are ones.
				static TensorFloat3D* Random(size_t* shape); // Creates a tensor with random values (0, 1). For test purposes.

															 // Access
				void Get(size_t* indices, float* out_value) const;
				void Set(size_t* indices, float in_value);
			};


			class TensorBinary : public Tensor {

			public:

				static TensorBinary* Create();
				static TensorBinary* Create(size_t dim, size_t* shape); // All values are zeros.
				static TensorBinary* Ones(size_t dim, size_t* shape); // All values are ones.
				static TensorBinary* Random(size_t dim, size_t* shape); // Creates a tensor with random values (0, 1). For test purposes.

																	   // Access
				void Get(size_t* indices, int* out_value) const;
				void Set(size_t* indices, int in_value);
			};

			class TensorBinary2D {

			public:

				static TensorBinary2D* Create();
				static TensorBinary2D* Create(size_t* shape); // All values are zeros.
				static TensorBinary2D* Ones(size_t* shape); // All values are ones.
				static TensorBinary2D* Random(size_t* shape); // Creates a tensor with random values (0, 1). For test purposes.

															 // Access
				void Get(size_t* indices, float* out_value) const;
				void Set(size_t* indices, float in_value);
			};

			class TensorBinary3D {

			public:

				static TensorBinary3D* Create();
				static TensorBinary3D* Create(size_t* shape); // All values are zeros.
				static TensorBinary3D* Ones(size_t* shape); // All values are ones.
				static TensorBinary3D* Random(size_t* shape); // Creates a tensor with random values (0, 1). For test purposes.

															 // Access
				void Get(size_t* indices, float* out_value) const;
				void Set(size_t* indices, float in_value);
			};*/

		} // tensor

		/*namespace layers {

			struct LayerDescriptor {
				void* parameters;
				char* layer_name;
			};

		} // functions*/


		// Model

		/*namespace model {

			struct SequentialModel {

				std::vector<layers::LayerDescriptor*> layer_descrs;
				unsigned char is_built; // unsigned char instead of bool
			};

			CORE_EXPORT extern nnsharp_status_p create_sequantial_model(SequentialModel* model);
			CORE_EXPORT extern nnsharp_status_p add_layer(SequentialModel* model, layers::LayerDescriptor* layer);
			CORE_EXPORT extern nnsharp_status_p build(SequentialModel* model);
			CORE_EXPORT extern nnsharp_status_p predict(SequentialModel model, tensor::Tensor* input, tensor::Tensor* output);

		} // model*/

	} // core

#endif // __NNSHARP__