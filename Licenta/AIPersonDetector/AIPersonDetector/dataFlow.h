#pragma once
#include "defines.h"
#include "neuralNetwork.h"
#include "dataFetcher.h"
#include "threadFunc.h"

class dataFlow
{
public:
	dataFlow();
	static void runNetImage(neuralNetwork & net, const unsigned &width, const unsigned &height, const string & inputImagesFile, const unsigned &imgNumber);
	static void runNetImageA(neuralNetwork & net, const unsigned &width, const unsigned &height, const string & inputImagesFile, const unsigned &imgNumber);
	static int runNetVideo(neuralNetwork &net, const unsigned &width, const unsigned &height, const string &filePath);
	static void trainNetImage(neuralNetwork & net, const unsigned & width, const unsigned & height, const string & trainingFilePath, const unsigned &imgNumber, const unsigned &epoch);
private:
	static void showResult(vector<double> inp);
	static unsigned getClass(vector<double> out);
	static void writeLog(vector<double> loggedVals);
};