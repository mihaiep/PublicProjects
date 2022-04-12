#include "dataAugm.h"
#include <math.h>

#define PI 3.141592653589793238462643383279502884

dataAugm::dataAugm()
{
}

cv::Mat dataAugm::_RTS(cv::Mat image, double rotAngleStep, double tranXStep, double tranYStep, double scale)
{
	cv::Mat nMat;
	// Scale up
	if (scale != 0)
	{
		cv::Rect corpImg(0, 0, image.cols, image.rows);
		cv::resize(image, nMat, cv::Size(), scale, scale, cv::INTER_LINEAR);
		nMat = nMat(corpImg);
	}
	else nMat = image;
	//Translation
	if (tranXStep != 0 || tranYStep!= 0)
	{
		cv::Mat trans = (cv::Mat_<double>(2, 3) << 1, 0, tranXStep, 0, 1, tranYStep);
		cv::warpAffine(image, nMat, trans, image.size());
	}
	//Rotation 
	double scaleR = 1.0;
	if (rotAngleStep != 0)
	{
		double angA = scaleR * cos(rotAngleStep * PI / 180);
		double angB = scaleR * sin(rotAngleStep * PI / 180);
		cv::Mat trans = (cv::Mat_<double>(2, 3) <<	 angA,	angB,	(1-angA) *(nMat.cols / 2.0) -    angB  * (nMat.rows / 2.0),
													-angB,	angA,	   angB	 *(nMat.cols / 2.0) + (1-angA) * (nMat.rows / 2.0));
		cv::warpAffine(image, nMat, trans, image.size());
	}
	return nMat;
}

cv::Mat dataAugm::_RGB(cv::Mat image, double rP, double gP, double bP)
{
	cv::Mat channels[3];
	cv::extractChannel(image, channels[0], 0);
	cv::extractChannel(image, channels[1], 1);
	cv::extractChannel(image, channels[2], 2);
	cv::waitKey();
	if (rP != 0)
	{
		channels[2] = channels[2] * (double)(1 + rP);
	}
	if (gP != 0)
	{
		channels[1] = channels[2] * (double)(1 + rP);
	}
	if (bP != 0)
	{
		channels[0] = channels[2] * (double)(1 + rP);
	}
	vector<cv::Mat> mergedV;
	mergedV.push_back(channels[0]);
	mergedV.push_back(channels[1]);
	mergedV.push_back(channels[2]);
	cv::Mat merged;
	cv::merge(mergedV, merged);	
	return merged;
}

cv::Mat dataAugm::_flipHorizontal(cv::Mat image)
{
	cv::Mat finalImg;
	cv::flip(image,finalImg,1);
	return finalImg;
}

cv::Mat dataAugm::_GaussianBlur(cv::Mat image, unsigned kernelSize)
{
	cv::Mat blurMat;
	cv::GaussianBlur(image, blurMat, cv::Size(kernelSize, kernelSize), 0);
	return blurMat;
}

cv::Mat dataAugm::_addNoise(cv::Mat image,unsigned min, unsigned max)
{
	cv::Mat saltPeper = cv::Mat::zeros(image.rows, image.cols, CV_8U);

	cv::randu(saltPeper, 0, 255);
	cv::Mat black = saltPeper < min;
	cv::Mat white = saltPeper > max;
	image.setTo(255, white);
	image.setTo(0, black);
	return image;
}