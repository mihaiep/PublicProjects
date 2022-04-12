#include "AIPersonDetector.h"
using namespace std;


//Chrono setup
std::chrono::system_clock::time_point start;
std::chrono::system_clock::time_point stop;

int main()
{
	srand(time(NULL));

	//Variables
	bool checkStructReady = false;
	bool checkNNinit = false;
	
	//string
	string inputInfo = "StructureCNN.txt";
	string trainingDataPath;

	//vectors
	vector<double> target;
	vector<nnStruct> myStructure;
	
	//Other objects
	kInfo cKernel, mKernel;
	dataFetcher convertor;
	neuralNetwork nNet;
	
	cout << "[Tema Licenta] Sistem inteligent de identificare a persoanelor" << endl;
	cout << "Autor: Mihai Eduard Petcu" << endl;
	cout << "Grupa: 443B" << endl << endl;

	short int operation;
	while (1)
	{
		showMenu();
		cin >> operation;
		cout << endl;
		if (operation == 0)
		{
			//Exit
			break;
		}
		else if (operation == 1)
		{			
			cout << endl << "[Incarcare arhitectura]" << endl;
			dataFetcher::setInputData(inputInfo, myStructure, cKernel, mKernel);
			cout << "Arhitectura Retea Incarcata!" << endl
				<< "Nuclee Incarcate!" << endl << endl;
			checkStructReady = true;
		}
		else if (operation == 2)
		{
			unsigned thNum;
			cout << endl << "[Initializare arhitectura retea]" << endl;
			cout << "Fire de executie [1;6]: ";
			cin >> thNum;
			cout << endl;
			if (thNum < 1) thNum = 1;
			if (thNum > 6) thNum = 6;

			getTimeStamp(start);
			nNet.netowkInit(myStructure, cKernel,mKernel,thNum);
			getTimeStamp(stop);
			cout << endl;
			coutTS(stop,start);
			cout << endl;
			checkNNinit = true;
			cout << "Structura a fost creata!" << endl << endl;
		}
		else if (operation == 3)
		{
			cout << endl << "[Antrenare]" << endl;
			if (checkStructReady == true && checkNNinit == true)
			{
				unsigned counter,epoci;
				string trainingFilePath;
				cin.ignore();

				cout << "Calea fisier training: " ;
				getline(cin, trainingFilePath, '\n');

				cout << "Numarul imagini: ";
				cin >> counter;

				cout << "Numar epoci: ";
				cin >> epoci;
				if (epoci != 1)
				{
					for (unsigned epN = 0; epN < epoci; epN++)
					{
						trainingFilePath[trainingFilePath.length() - 5] = char(49 + epN);
						dataFlow::trainNetImage(nNet, myStructure[0].width, myStructure[0].height, trainingFilePath, counter, epN + 1);
						nNet.resetEpochStats();
					}
				}
				else
				{
						dataFlow::trainNetImage(nNet, myStructure[0].width, myStructure[0].height, trainingFilePath, counter, 1);
				}

				cout << endl << "[Antrenare finalizata!]" << endl << endl;
			}
			else
			{
				cout << "Reteua nu a fost generata" << endl << endl;
			}
		}
		else if (operation == 4)
		{
			cout << endl << "[Rulare Retea]" << endl;
			if (checkStructReady == true && checkNNinit == true)
			{
				unsigned counter;
				string inputImagesFile;
				cin.ignore();

				cout << "Calea fisier lista imagini: ";
				getline(cin, inputImagesFile, '\n');

				cout << "Numarul imagini: ";
				cin >> counter;

				dataFlow::runNetImage(nNet, myStructure[0].width, myStructure[0].height, inputImagesFile, counter);

				cout << endl << "[Rulare finalizata!]" << endl << endl;
			}	
			else
			{
				cout << "Reteua nu a fost generata" << endl << endl;
			}
		}
		else if (operation == 5)
		{
			cout << endl << "[Rulare Retea Asistata]" << endl;
			if (checkStructReady == true && checkNNinit == true)
			{
				unsigned counter;
				string inputImagesFile;
				cin.ignore();

				cout << "Calea fisier lista imagini: ";
				getline(cin, inputImagesFile, '\n');

				cout << "Numarul imagini: ";
				cin >> counter;

				dataFlow::runNetImageA(nNet, myStructure[0].width, myStructure[0].height, inputImagesFile, counter);

				cout  << endl << "[Rulare finalizata!]" << endl << endl;
			}
			else
			{
				cout << "Reteua nu a fost generata" << endl << endl;
			}
		}
		else if (operation == 6)
		{
			//Importa Ponderi
			cout << endl << "[Importa ponderi]" << endl;
			if (checkStructReady == true && checkNNinit == true)
			{
				string filePath,
					fileName;
				cin.ignore();
				cout << "Cale fisier: ";
				getline(cin, filePath, '\n');

				nNet.importWeights(filePath);

				cout << endl << "Importare finalizata!" << endl << endl;

			}
			else
			{
				cout << "Reteua nu a fost generata" << endl << endl;
			}
		}
		else if (operation == 7)
		{
			//Salveaza Ponderi
			cout << endl << "[Salveaza ponderi]" << endl;
			if (checkStructReady == true && checkNNinit == true)
			{
				string filePath,
					fileName;

				cin.ignore();
				cout << "Cale folder: " ;
				getline(cin, filePath, '\n');

				cout << "Nume fisier: " ;
				getline(cin, fileName, '\n');

				pathCorrector(filePath);
				nNet.saveWeights(fileName, filePath);

				cout << endl << "Exportare finalizata!" << endl << endl;
			}
			else
			{
				cout << "Reteua nu a fost generata" << endl << endl;
			}
		}
		else if (operation == 8)
		{

			//Data Augmentation
			convertor.dataAugmentation();
		}
		else cout << "[Operatie invalida!] Selectati din nou!" << endl << endl;

	}
	system("PAUSE");
	return 0;
}