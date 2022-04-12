#pragma once
#include "Neuron.h"

struct EpochStat
{
	unsigned corretClass;
	unsigned generations;
};

class neuralNetwork
{
	public:
		neuralNetwork();
		void netowkInit(const vector<nnStruct> &myStructure, const kInfo &cKernel, const kInfo &mKernel, const unsigned &thNum);
		void feedForward(const vector<double>& inputs);
		void backProp(const vector<double> &targets);
		vector<double> getResults() const;
		double getAccuracy() const;
		double getError() const;
		void saveWeights(const string &fileName,const string &filePath) const;
		void importWeights(const string &filePath);
		void resetEpochStats();
	private:
		unsigned getClassNumber(vector<double> targets);
		vector<sLayer> nn_layers;
		double nn_error;
		double nn_accuracy;
		EpochStat zz_epStat;
		unsigned threadNum;		
};