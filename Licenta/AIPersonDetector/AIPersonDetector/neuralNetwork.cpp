#include "neuralNetwork.h"
#include "threadFunc.h"

// NeuralNetwork

neuralNetwork::neuralNetwork()
{

}

void neuralNetwork::netowkInit(const vector<nnStruct>& myStructure, const kInfo &cKernel, const kInfo &mKernel,const unsigned &thNum)
{
	this->threadNum = thNum;
	unsigned allNeurons = 0;
	unsigned nnSize = myStructure.size();
	//Create empty neurons
	for (unsigned nLayer = 0; nLayer < nnSize; nLayer++)
	{
		this->nn_layers.push_back(sLayer());
		this->nn_layers.back().layerStruct = myStructure[nLayer];

		//Input Layer
		//Generate neurons for all pixels H x W x D + 1 (bias)
		//and setting bias output on 1
		if (myStructure[nLayer].layerType == 'I')
		{

			this->nn_layers.back().Layer = vector<Neuron>(myStructure[nLayer].neuronNumber);
			//Bias
			this->nn_layers.back().Layer.push_back(Neuron(1.0));
		}

		//Conv Layers
		//Generate neurons: H x W x D + 1 (bias)
		//and setting bias output on 1
		else if (myStructure[nLayer].layerType == 'C')
		{
			this->nn_layers.back().Layer = vector<Neuron>(myStructure[nLayer].neuronNumber);
			if (myStructure[nLayer + 1].layerType != 'M')
			{
				this->nn_layers.back().Layer.push_back(Neuron(1.0));
			}
		}

		//MaxPooling Layers
		//Generate neurons: H x W x D + 1 (bias)
		//and setting bias output on 1
		else if (myStructure[nLayer].layerType == 'M')
		{
			this->nn_layers.back().Layer = vector<Neuron>(myStructure[nLayer].neuronNumber);
			this->nn_layers.back().Layer.push_back(Neuron(1.0));
		}

		//FullyConnected Layers
		//Generate neurons: H + 1 bias if is not last layer
		//Generate neurons: H if is last layer
		else if (myStructure[nLayer].layerType == 'F')
		{
			this->nn_layers.back().Layer = vector<Neuron>(myStructure[nLayer].neuronNumber);
			this->nn_layers.back().Layer.push_back(Neuron(1.0));
		}

		//Output Layer
		//Has no bias neuron
		else if (myStructure[nLayer].layerType == 'O')
		{
			for (unsigned nNeuron = 0; nNeuron < myStructure[nLayer].neuronNumber; nNeuron++)
			{
				this->nn_layers.back().Layer = vector<Neuron>(myStructure[nLayer].neuronNumber);
			}
		}
		allNeurons += this->nn_layers.back().Layer.size();
		cout<<"Stratul "<<nLayer<<" "<< this->nn_layers.back().Layer.size()<<endl;
	}

	cout << "Reteaua are: " << allNeurons << " neuroni." << endl;
	//Establish connections and set some variables for neurons
	for (unsigned nLayer = 1; nLayer < nnSize; nLayer++)
	{
		// Conv Layer
		if (myStructure[nLayer].layerType == 'C')
		{
			unsigned thIndex = 0;
			for (thIndex = 0; thIndex < this->threadNum - 1; thIndex++)
			{
				thread t(thread_SetNeuronC_MP, ref(this->nn_layers[nLayer]), ref(this->nn_layers[nLayer - 1]), cKernel, myStructure[nLayer],
					(double)this->nn_layers[nLayer].layerStruct.neuronNumber  / this->threadNum * thIndex, 
					(double)this->nn_layers[nLayer].layerStruct.neuronNumber  / this->threadNum * (thIndex + 1),
					(thIndex+1));
				t.detach();
			}
			unsigned lastN = this->nn_layers[nLayer].Layer.size();
			if (this->nn_layers[nLayer].Layer.back().Bias()) lastN --;
			thread t(thread_SetNeuronC_MP, ref(this->nn_layers[nLayer]), ref(this->nn_layers[nLayer - 1]), cKernel, myStructure[nLayer],
				(double)this->nn_layers[nLayer].layerStruct.neuronNumber / this->threadNum * thIndex,
				lastN,
				(thIndex + 1));
			t.detach();
			waitThreads(this->threadNum);
			resetWaitVals();
			cout << "Stratul " << nLayer << " " << myStructure[nLayer].layerType <<" a fost generat."<< endl;
		}

		//MaxPooling Layer
		else if (myStructure[nLayer].layerType == 'M')
		{
			unsigned thIndex = 0;
			for (thIndex = 0; thIndex < this->threadNum-1; thIndex++)
			{
				thread t(thread_SetNeuronC_MP, ref(this->nn_layers[nLayer]), ref(this->nn_layers[nLayer - 1]), mKernel, myStructure[nLayer], 
					(double)this->nn_layers[nLayer].layerStruct.neuronNumber / this->threadNum * thIndex,
					(double)this->nn_layers[nLayer].layerStruct.neuronNumber / this->threadNum * (thIndex + 1),
					(thIndex + 1));
				t.detach();
			}
			unsigned lastN = this->nn_layers[nLayer].Layer.size()-1;
			thread t(thread_SetNeuronC_MP, ref(this->nn_layers[nLayer]), ref(this->nn_layers[nLayer - 1]), mKernel, myStructure[nLayer],
				(double)this->nn_layers[nLayer].layerStruct.neuronNumber / this->threadNum * thIndex,
				lastN,
				(thIndex + 1));
			t.detach();
			waitThreads(this->threadNum);
			resetWaitVals();
			cout << "Stratul " << nLayer << " " << myStructure[nLayer].layerType << " a fost generat." << endl;
		}

		//FullyConnected Layer
		else if (myStructure[nLayer].layerType == 'F')
		{
			unsigned thIndex = 0;
			for (thIndex = 0; thIndex < this->threadNum-1; thIndex++)
			{
				thread t(thread_SetNeuron_F, ref(this->nn_layers[nLayer]), ref(this->nn_layers[nLayer - 1]), 
					(double)this->nn_layers[nLayer].layerStruct.neuronNumber / this->threadNum * thIndex,
					(double)this->nn_layers[nLayer].layerStruct.neuronNumber / this->threadNum * (thIndex + 1),
					(thIndex + 1));
				t.detach();
			}
			unsigned lastN = this->nn_layers[nLayer].Layer.size();
			if (this->nn_layers[nLayer].Layer.back().Bias()) lastN--;
			thread t(thread_SetNeuron_F, ref(this->nn_layers[nLayer]), ref(this->nn_layers[nLayer - 1]),
				(double)this->nn_layers[nLayer].layerStruct.neuronNumber / this->threadNum * thIndex,
				lastN,
				(thIndex + 1));
			t.detach();
			waitThreads(this->threadNum);
			resetWaitVals();
			cout << "Stratul " << nLayer << " " << myStructure[nLayer].layerType << " a fost generat." << endl;
		}

		//Output Layer
		else if (myStructure[nLayer].layerType == 'O')
		{
			unsigned num=this->nn_layers[nLayer].Layer.size();
			for (unsigned nNeuron = 0; nNeuron < myStructure[nLayer].neuronNumber; nNeuron++)
			{
				this->nn_layers.back().Layer[nNeuron].setFullyConNeuron(this->nn_layers[nLayer - 1]);
			}
			cout << "Stratul " << nLayer << " " << myStructure[nLayer].layerType << " a fost generat." << endl;
		}
	}
	cout << "Generarea retelei a fost terminata cu succes!" << endl;
	cout << "Se incepe initializarea ponderilor!" << endl;
	for (unsigned nLayer = 1; nLayer < nnSize; nLayer++)
	{
		if (this->nn_layers[nLayer].layerStruct.layerType != 'M')
		{
			unsigned lastN = this->nn_layers[nLayer].Layer.size();
			if (this->nn_layers[nLayer].Layer.back().Bias()) lastN--;
			if (this->nn_layers[nLayer].layerStruct.layerType != 'F' || this->nn_layers[nLayer].layerStruct.layerType != 'O')
			{
				for (unsigned nNeuron = 0; nNeuron < lastN; nNeuron++)
				{
					this->nn_layers[nLayer].Layer[nNeuron].randWH();
				}
			}
			else
			{
				for (unsigned nNeuron = 0; nNeuron < lastN; nNeuron++)
				{
					this->nn_layers[nLayer].Layer[nNeuron].randWF();
				}
			}
		}
	}
	cout << "Ponderile au fost generate cu succes!" << endl;
}

void neuralNetwork::feedForward(const vector<double>& inputs)
{
	if (inputs.size() != this->nn_layers[0].Layer.size()-1) exit(EXIT_FAILURE);

	this->zz_epStat.generations++;

	//Parallel input feed
	unsigned thIndex = 0;
	unsigned nNumber = this->nn_layers[0].Layer.size();
	if (this->nn_layers[0].Layer.back().Bias()) nNumber--;
	for (thIndex = 0; thIndex < this->threadNum-1; thIndex++)
	{
		thread t(thread_SetInputData_T, ref(this->nn_layers[0]), inputs, 
			(this->nn_layers[0].Layer.size()-1) / this->threadNum * thIndex,
			(this->nn_layers[0].Layer.size()-1) / this->threadNum * (thIndex+1),
			thIndex+1);
		t.detach();
	}
	thread t(thread_SetInputData_T , ref(this->nn_layers[0]), inputs,
		(this->nn_layers[0].Layer.size() - 1) / this->threadNum * thIndex,
		nNumber,
		thIndex + 1);
	t.detach();
	waitThreads(this->threadNum);
	resetWaitVals();

	//Parallel feedForwarding from layer 1 to last-1
	for (unsigned nLayer = 1; nLayer < this->nn_layers.size() - 1; nLayer++)
	{
		unsigned nNeuronMaxLayer=this->nn_layers[nLayer].Layer.size();
		if (this->nn_layers[nLayer].Layer.back().Bias()) nNeuronMaxLayer--;

		for (unsigned thIndex = 0; thIndex < this->threadNum-1; thIndex++)
		{
			thread t(thread_feedForward, ref(this->nn_layers[nLayer]), 
				this->nn_layers[nLayer].Layer.size() / this->threadNum * thIndex, 
				this->nn_layers[nLayer].Layer.size() / this->threadNum * (thIndex + 1), 
				thIndex + 1);
			t.detach();
		}
		thread t(thread_feedForward, ref(this->nn_layers[nLayer]),
			this->nn_layers[nLayer].Layer.size() / this->threadNum * (this->threadNum-1),
			nNeuronMaxLayer,
			this->threadNum);
		t.detach();
		waitThreads(this->threadNum);
		resetWaitVals();
	}
	//Calculate output last layer
	for (unsigned nNeuron = 0; nNeuron < this->nn_layers.back().Layer.size(); nNeuron++)
	{
		this->nn_layers.back().Layer[nNeuron].feedForwardO();
	}
}


void neuralNetwork::backProp(const vector<double>& targets)
{	
	
	//Error
	this->nn_error = 0.0;
	for (unsigned nOutput = 0; nOutput < this->nn_layers.back().Layer.size(); nOutput++)
	{
		this->nn_error += targets[nOutput] * log(this->nn_layers.back().Layer[nOutput].getOutput()) + (1 - targets[nOutput])*log(1 - this->nn_layers.back().Layer[nOutput].getOutput()); // cross-entropy
	}
	this->nn_error =-this->nn_error;
	
	//Acuratete
	vector<double> outVect;
	outVect = this->getResults();
	unsigned classNT = this->getClassNumber(targets),
		classNO = this->getClassNumber(outVect);
	if (classNT == classNO)
	{
		this->zz_epStat.corretClass++;
	}

	this->nn_accuracy = (double)this->zz_epStat.corretClass / this->zz_epStat.generations;
	// Calculate output layer gradients
	for (unsigned nNeuron = 0; nNeuron < this->nn_layers.back().Layer.size(); nNeuron++)
	{
		this->nn_layers.back().Layer[nNeuron].calcOutputGrad(targets[nNeuron]);
	}

	//Calculate gradients on hidden layers
	//Part 1. Calculate Neurons contributions
	//Part 2. Calculate Hiidengradient
	for (unsigned nLayer = this->nn_layers.size() - 2; nLayer > 0; nLayer--)
	{
		if (this->nn_layers[nLayer].layerStruct.layerType == 'M' && this->nn_layers[nLayer - 1].layerStruct.layerType == 'C')
		{
			unsigned thIndex = 0;
			unsigned nNumber = this->nn_layers[nLayer + 1].Layer.size();
			if (this->nn_layers[nLayer + 1].Layer.back().Bias()) nNumber--;
			for (thIndex = 0; thIndex < this->threadNum - 1; thIndex++)
			{
				thread t(thread_calcHiddenGradCont, ref(this->nn_layers[nLayer + 1]), ref(this->nn_layers[nLayer]),
					this->nn_layers[nLayer + 1].Layer.size() / this->threadNum * thIndex,
					this->nn_layers[nLayer + 1].Layer.size() / this->threadNum * (thIndex + 1),
					thIndex + 1);
				t.detach();
			}
			thread t(thread_calcHiddenGradCont, ref(this->nn_layers[nLayer + 1]), ref(this->nn_layers[nLayer]),
				this->nn_layers[nLayer + 1].Layer.size() / this->threadNum * thIndex,
				nNumber,
				thIndex + 1);
			t.detach();
			waitThreads(this->threadNum);
			resetWaitVals();

			thIndex = 0;
			nNumber = this->nn_layers[nLayer].Layer.size();
			for (thIndex = 0; thIndex < this->threadNum - 1; thIndex++)
			{
				thread t(thread_calcHiddenGrad, ref(this->nn_layers[nLayer]),
					this->nn_layers[nLayer].Layer.size() / this->threadNum * thIndex,
					this->nn_layers[nLayer].Layer.size() / this->threadNum * (thIndex + 1),
					thIndex + 1);
				t.detach();
			}
			thread t1(thread_calcHiddenGrad, ref(this->nn_layers[nLayer]),
				this->nn_layers[nLayer].Layer.size() / this->threadNum * thIndex,
				nNumber,
				thIndex + 1);
			t1.detach();
			waitThreads(this->threadNum);
			resetWaitVals();

			thIndex = 0;
			for (thIndex = 0; thIndex < this->threadNum - 1; thIndex++)
			{
				thread t(thread_gradCopy, ref(this->nn_layers[nLayer]),
					this->nn_layers[nLayer].Layer.size() / this->threadNum * thIndex,
					this->nn_layers[nLayer].Layer.size() / this->threadNum * (thIndex + 1),
					thIndex + 1);
				t.detach();
			}
			thread t2(thread_gradCopy, ref(this->nn_layers[nLayer]),
				this->nn_layers[nLayer].Layer.size() / this->threadNum * thIndex,
				this->nn_layers[nLayer].Layer.size() - 1, // are bias
				thIndex + 1);
			t2.detach();
			waitThreads(this->threadNum);
			resetWaitVals();
			nLayer--;
		}
		else
		{
			unsigned thIndex = 0;
			unsigned nNumber = this->nn_layers[nLayer + 1].Layer.size();
			if (this->nn_layers[nLayer + 1].Layer.back().Bias()) nNumber--;
			for (thIndex = 0; thIndex < this->threadNum - 1; thIndex++)
			{
				thread t(thread_calcHiddenGradCont, ref(this->nn_layers[nLayer + 1]), ref(this->nn_layers[nLayer]),
					this->nn_layers[nLayer + 1].Layer.size() / this->threadNum * thIndex,
					this->nn_layers[nLayer + 1].Layer.size() / this->threadNum * (thIndex + 1),
					thIndex + 1);
				t.detach();
			}
			thread t(thread_calcHiddenGradCont, ref(this->nn_layers[nLayer + 1]), ref(this->nn_layers[nLayer]),
				this->nn_layers[nLayer + 1].Layer.size() / this->threadNum * thIndex,
				nNumber,
				thIndex + 1);
			t.detach();
			waitThreads(this->threadNum);
			resetWaitVals();

			thIndex = 0;
			nNumber = this->nn_layers[nLayer].Layer.size();
			for (thIndex = 0; thIndex < this->threadNum - 1; thIndex++)
			{
				thread t(thread_calcHiddenGrad, ref(this->nn_layers[nLayer]),
					this->nn_layers[nLayer].Layer.size() / this->threadNum * thIndex,
					this->nn_layers[nLayer].Layer.size() / this->threadNum * (thIndex + 1),
					thIndex + 1);
				t.detach();
			}
			thread t1(thread_calcHiddenGrad, ref(this->nn_layers[nLayer]),
				this->nn_layers[nLayer].Layer.size() / this->threadNum * thIndex,
				nNumber,
				thIndex + 1);
			t1.detach();
			waitThreads(this->threadNum);
			resetWaitVals();
		}
	}

	// Update weights
	for (unsigned nNeuron = 0; nNeuron < this->nn_layers.back().Layer.size(); nNeuron++)
		this->nn_layers.back().Layer[nNeuron].updateWeights(this->nn_layers.back().layerStruct);
	for (unsigned nLayer = this->nn_layers.size() - 2; nLayer > 0; nLayer--)
	{
		unsigned thIndex = 0;
		unsigned nNumber = this->nn_layers[nLayer].Layer.size();
		if (this->nn_layers[nLayer].Layer.back().Bias()) nNumber--;
		for (thIndex = 0; thIndex < this->threadNum - 1; thIndex++)
		{
			thread t(thread_updateWeights, ref(this->nn_layers[nLayer]),
				this->nn_layers[nLayer].Layer.size() / this->threadNum * thIndex,
				this->nn_layers[nLayer].Layer.size() / this->threadNum * (thIndex + 1),
				thIndex + 1);
			t.detach();
		}
		thread t(thread_updateWeights, ref(this->nn_layers[nLayer]),
			this->nn_layers[nLayer].Layer.size() / this->threadNum * thIndex,
			nNumber,
			thIndex + 1);
		t.detach();
		waitThreads(this->threadNum);
		resetWaitVals();
	}
				
}

vector<double> neuralNetwork::getResults() const
{
	vector<double> results;
	for (unsigned nNeuron = 0; nNeuron < this->nn_layers.back().Layer.size(); nNeuron++)
	{
		results.push_back(this->nn_layers.back().Layer[nNeuron].getOutput());
	}
	return results;
}

double neuralNetwork::getAccuracy() const
{
	return this->nn_accuracy;
}

double neuralNetwork::getError() const
{
	return this->nn_error;
}

void neuralNetwork::saveWeights(const string &fileName, const string &filePath) const
{
	ofstream _oFile;
	_oFile.open(filePath+fileName);
	if (_oFile.is_open())
	{
		for (unsigned nLayer = 0; nLayer < this->nn_layers.size(); nLayer++)
		{
			_oFile << "Structure " << this->nn_layers[nLayer].layerStruct.layerType << " " 
				<< this->nn_layers[nLayer].Layer.size() << endl;
		}
		_oFile << endl << "Weights" << endl;
		for (unsigned nLayer = this->nn_layers.size() - 1; nLayer > 0; nLayer--)
		{
			for (unsigned nNeuron = 0; nNeuron < this->nn_layers[nLayer].Layer.size(); nNeuron++)
			{
				vector<double> weights=this->nn_layers[nLayer].Layer[nNeuron].getWeights();
				if (weights.size() == 0) _oFile << endl;
				else 
				for (unsigned nWeights = 0; nWeights < weights.size(); nWeights++)
					{
						_oFile << weights[nWeights];
						if (nWeights != weights.size() - 1) _oFile << " ";
						else _oFile << endl;
					}
			}
		}
		_oFile.close();
	}
	else
	{
		exit(EXIT_FAILURE);
	}
}

void neuralNetwork::importWeights(const string &filePath)
{
	unsigned nLayer = 0;
	ifstream _inpFile;
	//Has full path + file name
	_inpFile.open(filePath);
	if (_inpFile.is_open())
	{
		while (!_inpFile.eof())
		{
			string temp;
			getline(_inpFile, temp);
			stringstream sStream(temp);
			sStream >> temp;
			if (temp == "Structure")
			{
				sStream >> temp;
				if (temp[0] != this->nn_layers[nLayer].layerStruct.layerType) break;
				unsigned nNumber;
				sStream >> nNumber;
				if (nNumber != this->nn_layers[nLayer++].Layer.size()) break;
			}
			else if (temp == "Weights")
			{
				for (nLayer = this->nn_layers.size() - 1; nLayer > 0; nLayer--)
				{
					for (unsigned nNeuron = 0; nNeuron < this->nn_layers[nLayer].Layer.size(); nNeuron++)
					{
						getline(_inpFile, temp);
						stringstream sStream(temp);
						double tmp;
						vector<double> weights;
						while (!sStream.eof())
						{
							sStream >> tmp;
							weights.push_back(tmp);
						}
						this->nn_layers[nLayer].Layer[nNeuron].setWeights(weights);
					}
				}
			}
		}
		_inpFile.close();
	}
	else
	{
		exit(EXIT_FAILURE);
	}
}

void neuralNetwork::resetEpochStats()
{
	this->zz_epStat.corretClass = 0;
	this->zz_epStat.generations = 0;
}

unsigned neuralNetwork::getClassNumber(vector<double> targets)
{
	double max;
	unsigned pos;
	if (targets.size() >= 1)
	{
		pos = 0;
		max = targets[0];
	}
	for (unsigned posN=1;posN<targets.size();posN++)
	{
		if (max < targets[posN])
		{
			max = targets[posN];
			pos = posN;
		}
	}
	return pos;
}