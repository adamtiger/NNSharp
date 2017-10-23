// TryCNTKcpp.cpp : Defines the entry point for the console application.
//

#include <vector>
#include <iostream>

#include <CNTKLibrary.h>

using namespace CNTK;
using namespace Microsoft::MSR::CNTK;

// region: CNTK MINI TEST

std::vector<std::vector<size_t>> GenerateOneHotSequences(const std::vector<size_t>& sequenceLengths, size_t dim)
{
	size_t numSequences = sequenceLengths.size();
	std::vector<std::vector<size_t>> oneHotSequences;
	for (size_t i = 0; i < numSequences; ++i)
	{
		std::vector<size_t> currentSequence(sequenceLengths[i]);
		for (size_t j = 0; j < sequenceLengths[i]; ++j)
		{
			size_t hotRowIndex = rand() % dim;
			currentSequence[j] = hotRowIndex;
		}

		oneHotSequences.push_back(std::move(currentSequence));
	}

	return oneHotSequences;
}

void CNTK_test() {

	// Creating the computation tree.
	NDShape shape({ 1 });
	auto xInput = InputVariable(NDShape({ 1 }), DataType::Float);
	auto yInput = InputVariable(NDShape({ 1 }), DataType::Float);
	auto output = Plus(xInput, yInput);

	// Creating the data
	ValuePtr v_x = Value::Create<float>(shape, GenerateOneHotSequences({ 1 }, 1), DeviceDescriptor::CPUDevice());
	ValuePtr v_y = Value::Create<float>(shape, GenerateOneHotSequences({ 1 }, 1), DeviceDescriptor::CPUDevice());

	std::unordered_map<Variable, ValuePtr> args = { { xInput, v_x },{ yInput, v_y } };
	std::unordered_map<Variable, ValuePtr> ou = { { output->Output(), nullptr } };

	output->Evaluate(args, ou);

	auto v_out = ou[output->Output()];
	std::vector<float> outputData(v_out->Shape().TotalSize());
	NDArrayViewPtr arrayOutput = MakeSharedObject<NDArrayView>(v_out->Shape(), outputData, false);
	arrayOutput->CopyFrom(*v_out->Data());


	for (int idx = 0; idx < 1; ++idx) {

		std::cout << outputData[idx] << std::endl;
	}
}

// endregion (CNTK MINI TEST)

int main()
{

	CNTK_test();

    return 0;
}

