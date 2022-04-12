#pragma once
#include "defines.h"

class Neuron;
struct Connection
{
	double weight;
	double deltaWeight;
	Neuron *nLink;
};
struct sLayer
{
	nnStruct layerStruct;
	vector<Neuron> Layer;
};


class Neuron
	{
	public:
		Neuron();
		Neuron(const double &biasVal);
		void setFullyConNeuron(sLayer &previousLayer);
		void setFullyConNeuron(sLayer &previousLayer,unsigned a);
		void setConvMaxPNeuron(sLayer &previousLayer, const unsigned &index, const kInfo &Kernel, const nnStruct &layerStruct);
		double getOutput() const;
		void setOutput(const double &val);
		void feedForward();
		void feedForwardO();
		void calcOutputGrad(const double &target);
		void calcHiddenGradCont(const nnStruct & layerStruct, const nnStruct & prevLayer);
		void calcHiddenGrad();
		double getGradient() const;
		void updateWeights(const nnStruct &currentStruct);
		void setWeights(const vector<double> &newWeight);
		void copyGrad();
		void randWF();
		void randWH();
		vector<double> getWeights() const;
		bool Bias();
	private:
		static double transferFunction(double x); 
		static double transferFunctionD(double x); 
		static double outFunc(double x);
		static double outFuncD(double x);
		static double randomWeight();
		static double randomFCWeight();
		double n_output;
		double n_gradient;		
		double _hiddenGradCont;  	
		vector <Connection> c_weights;
		double n_net;
		bool isMaxPooling;
		bool isBias;				
		static double eta;	
		static double alpha;
	};