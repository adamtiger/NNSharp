#ifndef __NNSHARP__
#define __NNSHARP__

#ifndef CORE_EXPORT
#	if defined COMPILER_MSVC
#		define CORE_EXPORT __declspec(dllexport)
#	else 
#		define CORE_EXPORT __attribute__((visibility("default")))
#	endif
#endif

#ifndef CORE_EXTERN_C
#	if defined __cplusplus
#		define CORE_EXTERN_C extern "C"
#	else
#		define CORE_EXTERN_C
#	endif
#endif

#if defined WIN32 || _WIN32
#	define CORE_CDECL __cdecl
#	define CORE_STDCALL __stdcall
#else
# define CORE_CDECL 
# define CORE_STDCALL
#endif

#ifndef CORE
#	define CORE(rettype) CORE_EXTERN_C CORE_EXPORT rettype CORE_CDECL
#endif

#include <vector>

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
				Tensor(DataType data_type, int dim, int* shape);

				~Tensor();

			protected:

				DataType data_type;
				void* values;
				int dim;
				int* shape;
			};

			class TensorInteger : public Tensor {

			public:
				TensorInteger(int size);

				// Access
				void Get(int idx, int* value_out) const;
				void Set(int idx, int value_in);
			};

			CORE(TensorInteger*) create_tensor_integer(int size);
			CORE(int) tensor_integer_get(TensorInteger* tensor_in, int idx);
		}

	} // core

#endif // __NNSHARP__