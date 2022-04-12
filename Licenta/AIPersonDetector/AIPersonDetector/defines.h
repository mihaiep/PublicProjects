#pragma once

//Libraries
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <iostream>
#include <vector>
#include <thread>
//#include <Windows.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

//Namespaces
using namespace std;

//Others
#ifndef _NN_STRUCT_
	#define _NN_STRUCT_
	struct nnStruct
	{
		unsigned neuronNumber;
		char layerType;
		unsigned depth;
		unsigned width;
		unsigned height;
	};
#endif // !_NN_STRUCT_

#ifndef _KINFO_STRUCT_
	#define _KINFO_STRUCT_
	struct kInfo
	{
		unsigned width;
		unsigned height;
		unsigned stride;
	};
#endif // !_KINFO_STRUCT_