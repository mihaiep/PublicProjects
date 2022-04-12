#include "dataFlow.h"
#include "AIPersonDetector.h"

bool inputNewFrameFF = false;

dataFlow::dataFlow()
{
}

void dataFlow::runNetImage(neuralNetwork & net, const unsigned &width, const unsigned &height, const string & inputImagesFile, const unsigned &imgNumber)
{
	dataFetcher dFetch;
	ifstream inpFile;
	inpFile.open(inputImagesFile);
	std::chrono::system_clock::time_point stop;
	std::chrono::system_clock::time_point start;
	if (inpFile.is_open())
	{
		unsigned goodGuess = 0;
		for (int fileN = 0; fileN < imgNumber && !inpFile.eof(); fileN++)
		{
			cout << endl << fileN + 1 << ": " << endl;
			string fileName, targetVects;
			getline(inpFile, fileName, ';');
			getline(inpFile, targetVects, '\n');
			dFetch.setObjects(fileName, width, height);
			getTimeStamp(start);
			net.feedForward(dFetch.getVector());
			getTimeStamp(stop);
			vector<double> results = net.getResults();
			vector<double> targets;
			stringstream targetStream(targetVects);
			while (!targetStream.eof())
			{
				double tmp;
				targetStream >> tmp;
				targets.push_back(tmp);
			}			
			cout << "Output: " << endl;
			dataFlow::showResult(results);
			cout << endl;
			cout << "Target: " << endl;
			dataFlow::showResult(targets);
			cout << endl;
			if (getClass(targets) == getClass(results)) goodGuess++;
			cout << "Acuratete: " << (double)goodGuess / (fileN + 1) * 100 << "%" << endl << endl;
			coutTS(stop, start);
			dataFlow::writeLog(results);
			dFetch.clearObj();
		}
	}
	cout << endl;
}

void dataFlow::runNetImageA(neuralNetwork & net, const unsigned & width, const unsigned & height, const string & inputImagesFile, const unsigned & imgNumber)
{
	dataFetcher dFetch;
	ifstream inpFile;
	std::chrono::system_clock::time_point stop;
	std::chrono::system_clock::time_point start;
	inpFile.open(inputImagesFile);
	if (inpFile.is_open())
	{
		unsigned goodGuess = 0;
		for (int fileN = 0; fileN < imgNumber && !inpFile.eof(); fileN++)
		{
			cout << endl << fileN + 1 << ": " << endl;
			string fileName;
			getline(inpFile, fileName, ';');
			dFetch.setObjects(fileName, width, height);
			getline(inpFile, fileName, '\n');
			getTimeStamp(start);
			net.feedForward(dFetch.getVector());
			getTimeStamp(stop);
			vector<double> results = net.getResults();
			cout << "Output: " << endl;
			dataFlow::showResult(results);
			cout << endl;
			cv::imshow("Img", dFetch.getDFImage());
			dataFlow::writeLog(results);
			coutTS(stop, start);
			cv::waitKey();
			dFetch.clearObj();
		}
	}
	cout << endl;
}

int dataFlow::runNetVideo(neuralNetwork &net, const unsigned &width, const unsigned &height, const string &filePath)
{
	cv::VideoCapture video(filePath);
	if (video.isOpened())
	{
		dataFetcher dFetch;
		cv::Mat temp;
		while (1)
		{
			if (!video.grab()) return -1;
			else
			{			
				video.retrieve(temp);
				dFetch.setObjects(temp, width, height);
				vector<double> inputs = dFetch.getVector();
				thread t(thread_runNet, ref(net), ref(inputs));
				t.detach();
				while (!inputNewFrameFF) 
				{					
					if (!video.grab()) return -1;
					video.retrieve(temp);
				}
				dFetch.clearObj();
				inputNewFrameFF = false;
			}
		}
	}
	else
	{
		cout << "Fisierul nu a putut fi deschis." << endl;
	}
}

void dataFlow::trainNetImage(neuralNetwork & net, const unsigned & width, const unsigned & height, const string & trainingFilePath, const unsigned &imgNumber, const unsigned &epoch)
{
	
	ifstream trainingFile;
	trainingFile.open(trainingFilePath);
	if (trainingFile.is_open())
	{
		cout << endl;
		for (int fileN = 0; fileN < imgNumber && !trainingFile.eof(); fileN++)
		{
			dataFetcher dFetch;
			string fileName, targetVects;
			getline(trainingFile, fileName, ';');
			getline(trainingFile, targetVects, '\n');
			dFetch.setObjects(fileName, width, height);
			net.feedForward(dFetch.getVector());
			vector<double> results = net.getResults();			
			vector<double> targets;
			stringstream targetStream(targetVects);
			while (!targetStream.eof())
			{
				double tmp;
				targetStream >> tmp;
				targets.push_back(tmp);
			}
			net.backProp(targets);
			cout << "E" << epoch << "I" << fileN + 1 << endl;
			cout << "Output: " << endl;
			dataFlow::showResult(results);		
			cout << endl;
			cout << "Target: " << endl;
			dataFlow::showResult(targets);
			cout << endl;
			cout << "Clasa " << dataFlow::getClass(results) << " - " << dataFlow::getClass(targets) << " Target" << endl << "Eroare: " << net.getError() * 100 << "%" << endl << "Acuratete: " << net.getAccuracy() * 100 << "%";
			cout << endl << "--------------------------------------------------------" << endl;
		}
	}
}

void dataFlow::showResult(vector<double> inp)
{
	for (unsigned resN = 0; resN < inp.size(); resN++)
	{
		if (resN == 0)
		{
			cout << "[ ";
			cout << setprecision(3) << setw(9) << inp[resN];
		}
		else if (resN == inp.size() - 1)
		{
			cout << " | ";
			cout << setprecision(3) << setw(9) << inp[resN];
			cout << "]";
		}
		else
		{
			cout << " | ";
			cout << setprecision(3) << setw(9) << inp[resN];
		}
	}
}

unsigned dataFlow::getClass(vector<double> out)
{
	unsigned classNum = 0;
	double max=out[0];
	for (unsigned it = 1; it < out.size(); it++)
	{
		if (max < out[it])
		{
			classNum = it;
			max = out[it];
		}
	}
	return classNum + 1;
}

void dataFlow::writeLog(vector<double> loggedVals)
{
	ofstream outLogFile;
	outLogFile.open("log.csv", ios::app);
	for (unsigned len = 0; len < loggedVals.size(); len++)
	{
		if (len != loggedVals.size() - 1) outLogFile << loggedVals[len] << ";";
		else outLogFile << loggedVals[len] << endl;
	}

}
