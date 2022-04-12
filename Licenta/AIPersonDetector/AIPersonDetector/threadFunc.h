#pragma once
#include "neuralNetwork.h"


// Thread Mangement Function 
void waitThreads(const unsigned &thNum);
void resetWaitVals();

//Thread Utility Functions

void thread_SetNeuronC_MP		(sLayer &currentLayer,	sLayer &prevLayer, const kInfo Kernel, const nnStruct layerStruct,	unsigned startNeuron, unsigned endNeuron, unsigned short thId);
void thread_SetNeuron_F			(sLayer &currentLayer,	sLayer &prevLayer,													unsigned startNeuron, unsigned endNeuron, unsigned short thId);
void thread_SetInputData_T		(sLayer &currentLayer,	vector<double> inputs,												unsigned startNeuron, unsigned endNeuron, unsigned short thId);
void thread_feedForward			(sLayer &currentLayer,																		unsigned startNeuron, unsigned endNeuron, unsigned short thId);
void thread_calcHiddenGradCont	(sLayer & currentLayer, sLayer & prevLayer,													unsigned startNeuron, unsigned endNeuron, unsigned short thId);
void thread_calcHiddenGrad		(sLayer & currentLayer,																		unsigned startNeuron, unsigned endNeuron, unsigned short thId);
void thread_updateWeights		(sLayer & currentLayer,																		unsigned startNeuron, unsigned endNeuron, unsigned short thId);
void thread_gradCopy			(sLayer & currentLayer,																		unsigned startNeuron, unsigned endNeuron, unsigned short thId);
void thread_runNet				(neuralNetwork &net, vector<double> &inputs);