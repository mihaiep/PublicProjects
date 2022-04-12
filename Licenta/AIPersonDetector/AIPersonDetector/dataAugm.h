#pragma once
#include "defines.h"
class dataAugm
{
public:
	dataAugm();
	static cv::Mat _RTS(cv::Mat image, double rotAngleStep, double tranXStep, double tranYStep, double scale); //Rotation Translation Scalling
	static cv::Mat _RGB(cv::Mat image, double rP, double gP, double bP); //RGB modification
	static cv::Mat _flipHorizontal(cv::Mat image);
	static cv::Mat _GaussianBlur(cv::Mat image, unsigned kernelSize);
	static cv::Mat _addNoise(cv::Mat image, unsigned min, unsigned max);
};