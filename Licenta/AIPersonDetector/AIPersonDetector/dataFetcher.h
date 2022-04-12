#pragma once
#include "defines.h"
#include "dataAugm.h"

class dataFetcher
{
public:
	dataFetcher();
	vector<double> getVector();
	cv::Mat getDFImage();
	void setObjects(const string &Path, const double & width, const double & height); // Image Path, resize dims(width,height), normalization interval[minVal,maxVal]
	void setObjects(cv::Mat img, const double & width, const double & height);
	static void setInputData(string filePath, vector<nnStruct> &myStructure, kInfo &cKernel, kInfo &mKernel);
	void clearObj();
	void dataAugmentation();
private:
	cv::Mat getImage(string path);
	string formatName(string name, unsigned number);
	string formatName(string name, string tr, unsigned number);
	string formatName(string name, string tr);
	string formatName(string name,double R, double G, double B);
	string formatName(string name, double R, double G, double B, unsigned number);
	string formatName(string name, double R, double Tx, double Ty, double S);
	string formatName(string name, double R, double Tx, double Ty, double S, unsigned number);
	void imagesSave(const string & path, const string &name, vector<cv::Mat> vectImg);
	void resizeImage(const double &width, const double &height);
	void normalizeVector();
	void setVector();
	string getFolderName(string fileName);
	vector<double> imgVector;
	cv::Mat image;
};