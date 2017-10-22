// TryCNTKcpp.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include <vector>
#include <iostream>

//region: TF MINI TEST

#define COMPILER_MSVC
#include "tensorflow.h"
#include "nnsharp.h"

void tf_test() {

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
	tf_test();
	

    return 0;
}

