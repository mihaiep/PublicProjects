#include "Neuron.h"


double Neuron::alpha = 0.001;  // [0.0 .. n] momentum
double Neuron::eta = 0.01;	  // [0.0 .. 1.0] overall net training rate


//Input Layer
Neuron::Neuron()
{
	this->isBias = false;
	this->isMaxPooling = false;
	this->_hiddenGradCont = 0;
	this->n_output = 0;
	this->n_gradient = 0;
}

//Bias
Neuron::Neuron(const double & biasVal)
{
	this->isMaxPooling = false;
	this->_hiddenGradCont = 0;
	this->n_output = biasVal;
	this->isBias = true;
}

//FullyConnected
void Neuron::setFullyConNeuron(sLayer & previousLayer)
{
	for (unsigned nNeuron = 0; nNeuron < previousLayer.Layer.size(); nNeuron++)
	{
		this->c_weights.push_back(Connection());
		this->c_weights.back().nLink = &previousLayer.Layer[nNeuron];
	}
}

void Neuron::setFullyConNeuron(sLayer & previousLayer,unsigned a)
{
	if (a == 0)
	{
		for (unsigned nNeuron = 0; nNeuron < (previousLayer.Layer.size()-1)/2; nNeuron++)
		{
			this->c_weights.push_back(Connection());
			this->c_weights.back().nLink = &previousLayer.Layer[nNeuron];
		}
		this->c_weights.push_back(Connection());
		this->c_weights.back().weight = randomFCWeight();
		this->c_weights.back().nLink = &previousLayer.Layer.back();
	}
	else
	{
		for (unsigned nNeuron = (previousLayer.Layer.size() - 1) / 2; nNeuron < previousLayer.Layer.size(); nNeuron++)
		{
			this->c_weights.push_back(Connection());
			this->c_weights.back().nLink = &previousLayer.Layer[nNeuron];
		}
	}
}

//Hidden Layers with Kernel
void Neuron::setConvMaxPNeuron(sLayer &previousLayer, const unsigned &index, const kInfo &Kernel, const nnStruct &layerStruct)
{
	unsigned prevH=previousLayer.layerStruct.height
		, prevW=previousLayer.layerStruct.width
		, prevD=previousLayer.layerStruct.depth;
	if (index < layerStruct.depth*layerStruct.height*layerStruct.width)
	{
		unsigned _offset;
		//ConvLayer
		if (layerStruct.layerType == 'C')
		{
			_offset = ((index % (layerStruct.height*layerStruct.width)) / layerStruct.width) * prevW		//row offset
				+ index % layerStruct.width;																//column offset
			this->isMaxPooling = false;
			for (unsigned nLayer = 0; nLayer < prevD; nLayer++)
			{
				for (unsigned nRow = 0; nRow < Kernel.height; nRow++)
				{
					for (unsigned nCol = 0; nCol < Kernel.width; nCol++)
					{
						this->c_weights.push_back(Connection());
						this->c_weights.back().nLink = &previousLayer.Layer[nLayer*prevH*prevW + nRow * prevW + nCol + _offset];
					}
				}
			}
			//Connect to Bias from prev Layer
			if (previousLayer.Layer.back().Bias())
			{
				this->c_weights.push_back(Connection());
				this->c_weights.back().nLink = &previousLayer.Layer.back();
			}
		}

		//MaxPoolingLayer
		else if (layerStruct.layerType == 'M')
		{
			_offset = (index / (layerStruct.height*layerStruct.width)*prevH*prevW)								//layer offset
				+ (index % (layerStruct.width*layerStruct.height) / layerStruct.width)*prevW*Kernel.stride		//row offset
				+ (index%layerStruct.width) *Kernel.stride;														//column offset
			this->isMaxPooling = true;
			for (unsigned nRow = 0; nRow < Kernel.height; nRow++)
			{
				for (unsigned nCol = 0; nCol < Kernel.width; nCol++)
				{
					if (_offset % prevW + nCol >= prevW || (_offset % (prevH*prevW) + nRow * prevW) / (prevH*prevW) >= 1) continue;
					this->c_weights.push_back(Connection());
					this->c_weights.back().nLink = &previousLayer.Layer[nRow*prevW + nCol + _offset];
					this->c_weights.back().weight = 0.0;
				}
			}
		}
	}
}

double Neuron::getOutput() const
{
	return this->n_output;
}

void Neuron::setOutput(const double &val)
{
	this->n_gradient = 0.0;
	this->_hiddenGradCont = 0.0;
	this->n_output = val;
}

void Neuron::feedForward()
{
	this->_hiddenGradCont = 0.0;
	this->n_gradient = 0.0;
	if (!this->isMaxPooling)
	{
		double net = 0.0;
		for (unsigned conNum = 0; conNum < this->c_weights.size(); conNum++)
		{
			net += this->c_weights[conNum].weight * this->c_weights[conNum].nLink->getOutput();
		}
		this->n_net = net;
		this->n_output = transferFunction(n_net);
	}
	else 
	{
		//MaxPooling
		//Special feedForward. picking biggest values from its connections
		this->n_output = this->c_weights[0].nLink->getOutput();
		this->n_net = this->c_weights[0].nLink->n_net;
		for (unsigned conNum = 1; conNum < this->c_weights.size(); conNum++)
		{
			if (this->c_weights[conNum].nLink->getOutput() > this->n_output)
			{
				this->n_output = this->c_weights[conNum].nLink->getOutput();
				this->n_net = this->n_output;
			}
		}
	}	
}

void Neuron::feedForwardO()
{
	this->_hiddenGradCont = 0;
	this->n_gradient = 0;
	double net = 0.0;
	for (unsigned conN = 0; conN < this->c_weights.size(); conN++)
	{
		net += this->c_weights[conN].weight*this->c_weights[conN].nLink->getOutput();
	}
	this->n_net = net;
	cout << "Net: " << net << endl;
	this->n_output = outFunc(this->n_net);
}


void Neuron::calcOutputGrad(const double & target)
{
	this->n_gradient = (target - this->n_output);
}


void Neuron::calcHiddenGradCont(const nnStruct & layerStruct, const nnStruct & prevLayer)
{
	//	F-M-C <=> F-C
	//	C-M-C <=> C-C	
	// Contributions are calculated for F/C neuron on C neurons bypassing the MaxPooling Layer
	if (!this->isBias)
	{
		unsigned conNumber;
		conNumber = this->c_weights.size();
		for (unsigned nCon = 0; nCon < conNumber; nCon++)
		{
			this->c_weights[nCon].nLink->_hiddenGradCont += this->c_weights[nCon].weight*this->n_gradient;
		}
	}	
}

void Neuron::calcHiddenGrad()
{
	this->n_gradient = this->_hiddenGradCont*this->transferFunctionD(this->n_net);
}

double Neuron::getGradient() const
{
	return this->n_gradient;
}

void Neuron::updateWeights(const nnStruct &currentStruct)
{
	if (currentStruct.layerType != 'M')
	{
		for (unsigned nCon = 0; nCon < this->c_weights.size(); nCon++)
		{
			double oldDW = this->c_weights[nCon].deltaWeight;
			double newDW = this->eta * this->c_weights[nCon].nLink->getOutput() * this->getGradient() + this->alpha * oldDW;
			this->c_weights[nCon].deltaWeight = newDW;
			this->c_weights[nCon].weight += newDW;
			
		}
	}
}

void Neuron::setWeights(const vector<double> &newWeight)
{
	for (unsigned nCon = 0; nCon < this->c_weights.size(); nCon++)
	{
		this->c_weights[nCon].weight = newWeight[nCon];
	}
}

void Neuron::copyGrad()
{
	for (unsigned nCon = 0; nCon < this->c_weights.size(); nCon++)
	{
		if (this->n_output == this->c_weights[nCon].nLink->getOutput())
		{
			this->c_weights[nCon].nLink->n_gradient = this->n_gradient;
		}
	}
}

vector<double> Neuron::getWeights() const
{
	vector<double> weights;
	for (unsigned nCon = 0; nCon < this->c_weights.size(); nCon++)
	{
		weights.push_back(this->c_weights[nCon].weight);
	}
	return weights;
}
bool Neuron::Bias()
{
	if (this->isBias) return true;
	return false;
}

double Neuron::transferFunction(double x)
{
	//relu
	if (x > 0)return x;
	return 0;
}

double Neuron::transferFunctionD(double x)
{
	//relu
	if (x > 0) return 1.0;
	return 0;
}

double Neuron::outFunc(double x)
{
	//sigmoida
	return (double)1 / (1 + exp(-x)); 
}

double Neuron::outFuncD(double x)
{
	//derivata sigmoida
	return (double)outFunc(x) * (1 - outFunc(x));
}


double Neuron::randomWeight()
{
	return (double)(rand() - (RAND_MAX / 2)) / (RAND_MAX / 2)/8;
}

double Neuron::randomFCWeight()
{
	return (double)(rand()) / RAND_MAX / 10;
}

void Neuron::randWH()
{
	for (unsigned nCon = 0; nCon < this->c_weights.size(); nCon++)
	{
		this->c_weights[nCon].weight = randomWeight();
	}
}

void Neuron::randWF()
{
	for (unsigned nCon = 0; nCon < this->c_weights.size(); nCon++)
	{
		this->c_weights[nCon].weight = randomFCWeight();
	}
}