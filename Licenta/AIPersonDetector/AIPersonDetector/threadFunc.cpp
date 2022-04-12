#include "threadFunc.h"

extern bool inputNewFrameFF;

bool waitThread1 = false;
bool waitThread2 = false;
bool waitThread3 = false;
bool waitThread4 = false;
bool waitThread5 = false;
bool waitThread6 = false;

void waitThreads(const unsigned &thNum)
{	
			if (thNum == 1) while (!waitThread1);
	else	if (thNum == 2)	while (!waitThread1 || !waitThread2);
	else	if (thNum == 3)	while (!waitThread1 || !waitThread2 || !waitThread3);
	else	if (thNum == 4)	while (!waitThread1 || !waitThread2 || !waitThread3 || !waitThread4);
	else	if (thNum == 5)	while (!waitThread1 || !waitThread2 || !waitThread3 || !waitThread4 || !waitThread5);
	else	if (thNum == 6)	while (!waitThread1 || !waitThread2 || !waitThread3 || !waitThread4 || !waitThread5 || !waitThread6);
}

void resetWaitVals()
{
	waitThread1 = false;
	waitThread2 = false;
	waitThread3 = false;
	waitThread4 = false;
	waitThread5 = false;
	waitThread6 = false;
}

void thread_SetNeuronC_MP(sLayer &currentLayer, sLayer &prevLayer, const kInfo Kernel, const nnStruct layerStruct,
	unsigned startNeuron, unsigned endNeuron, unsigned short thId)
{

	for (unsigned nNeuron = startNeuron; nNeuron < endNeuron; nNeuron++)
		currentLayer.Layer[nNeuron].setConvMaxPNeuron(prevLayer, nNeuron, Kernel, layerStruct);
	if (thId == 1) waitThread1 = true;
	else if (thId == 2) waitThread2 = true;
	else if (thId == 3) waitThread3 = true;
	else if (thId == 4) waitThread4 = true;
	else if (thId == 5) waitThread5 = true;
	else if (thId == 6) waitThread6 = true;
}

void thread_SetNeuron_F(sLayer &currentLayer, sLayer &prevLayer,
	unsigned startNeuron, unsigned endNeuron, unsigned short thId)
{
	for (unsigned nNeuron = startNeuron; nNeuron < endNeuron; nNeuron++)
			currentLayer.Layer[nNeuron].setFullyConNeuron(prevLayer);
	if (thId == 1) waitThread1 = true;
	else if (thId == 2) waitThread2 = true;
	else if (thId == 3) waitThread3 = true;
	else if (thId == 4) waitThread4 = true;
	else if (thId == 5) waitThread5 = true;
	else if (thId == 6) waitThread6 = true;
}

void thread_SetInputData_T(sLayer &currentLayer, vector<double> inputs,
	unsigned startNeuron, unsigned endNeuron, unsigned short thId)
{
	for (unsigned nNeuron = startNeuron; nNeuron < endNeuron; nNeuron++)
		currentLayer.Layer[nNeuron].setOutput(inputs[nNeuron]);
	if (thId == 1) waitThread1 = true;
	else if (thId == 2) waitThread2 = true;
	else if (thId == 3) waitThread3 = true;
	else if (thId == 4) waitThread4 = true;
	else if (thId == 5) waitThread5 = true;
	else if (thId == 6) waitThread6 = true;
}

void thread_feedForward(sLayer &currentLayer,
	unsigned startNeuron, unsigned endNeuron, unsigned short thId)
{
	for (unsigned nNeuron = startNeuron; nNeuron < endNeuron; nNeuron++)
		currentLayer.Layer[nNeuron].feedForward();
	if (thId == 1) waitThread1 = true;
	else if (thId == 2) waitThread2 = true;
	else if (thId == 3) waitThread3 = true;
	else if (thId == 4) waitThread4 = true;
	else if (thId == 5) waitThread5 = true;
	else if (thId == 6) waitThread6 = true;
}

void thread_calcHiddenGradCont(sLayer & currentLayer, sLayer & prevLayer, unsigned startNeuron, unsigned endNeuron, unsigned short thId)
{
	for (unsigned nNeuron = startNeuron; nNeuron < endNeuron; nNeuron++)
		currentLayer.Layer[nNeuron].calcHiddenGradCont(currentLayer.layerStruct, prevLayer.layerStruct);
	if (thId == 1) waitThread1 = true;
	else if (thId == 2) waitThread2 = true;
	else if (thId == 3) waitThread3 = true;
	else if (thId == 4) waitThread4 = true;
	else if (thId == 5) waitThread5 = true;
	else if (thId == 6) waitThread6 = true;
}

void thread_calcHiddenGrad(sLayer & currentLayer, unsigned startNeuron, unsigned endNeuron, unsigned short thId)
{
	for (unsigned nNeuron = startNeuron; nNeuron < endNeuron; nNeuron++)
	{
		currentLayer.Layer[nNeuron].calcHiddenGrad();
	}
	if (thId == 1) waitThread1 = true;
	else if (thId == 2) waitThread2 = true;
	else if (thId == 3) waitThread3 = true;
	else if (thId == 4) waitThread4 = true;
	else if (thId == 5) waitThread5 = true;
	else if (thId == 6) waitThread6 = true;
}

void thread_updateWeights(sLayer & currentLayer, unsigned startNeuron, unsigned endNeuron, unsigned short thId)
{
	for (unsigned nNeuron = startNeuron; nNeuron < endNeuron; nNeuron++)
	{
		currentLayer.Layer[nNeuron].updateWeights(currentLayer.layerStruct);
	}
	if (thId == 1) waitThread1 = true;
	else if (thId == 2) waitThread2 = true;
	else if (thId == 3) waitThread3 = true;
	else if (thId == 4) waitThread4 = true;
	else if (thId == 5) waitThread5 = true;
	else if (thId == 6) waitThread6 = true;
}

void thread_gradCopy(sLayer & currentLayer, unsigned startNeuron, unsigned endNeuron, unsigned short thId)
{
	for (unsigned nNeuron = startNeuron; nNeuron < endNeuron; nNeuron++)
		currentLayer.Layer[nNeuron].copyGrad();
	if (thId == 1) waitThread1 = true;
	else if (thId == 2) waitThread2 = true;
	else if (thId == 3) waitThread3 = true;
	else if (thId == 4) waitThread4 = true;
	else if (thId == 5) waitThread5 = true;
	else if (thId == 6) waitThread6 = true;
}

void thread_runNet(neuralNetwork & net, vector<double>& inputs)
{
	net.feedForward(inputs);
	inputNewFrameFF = true;
	vector<double> results = net.getResults();
	for (unsigned resN = 0; resN < results.size(); resN++)
	{
		cout << "Neuron " << resN + 1 << " :" << results[resN] << endl;
	}
}