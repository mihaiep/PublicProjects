#include "dataFetcher.h"
#include "defines.h"


dataFetcher::dataFetcher()
{
}

vector<double> dataFetcher::getVector()
{
	return this->imgVector;
}

cv::Mat dataFetcher::getDFImage()
{
	return this->image;
}

void dataFetcher::setObjects(const string &Path, const double & width, const double & height)
{
	this->image = cv::imread(Path, cv::IMREAD_COLOR);
	this->resizeImage(width,height);
	this->setVector();
}

void dataFetcher::setObjects(cv::Mat img, const double & width, const double & height)
{
	this->image = img;
	this->resizeImage(width, height);
	this->setVector();
	this->normalizeVector();
}

void dataFetcher::setInputData(string filePath, vector<nnStruct> &myStructure, kInfo &cKernel, kInfo &mKernel)
{
	ifstream input;
	input.open(filePath);
	if (input.is_open())
	{
		string word;
		stringstream ss;
		ss << input.rdbuf();
		while (!ss.eof())
		{
			ss >> word;
			unsigned temp;
			if (word == "KC:")
			{
				ss >> temp;
				cKernel.width = temp;
				ss >> temp;
				cKernel.height = temp;
				ss >> temp;
				cKernel.stride = temp;
			}
			else if (word == "KM:")
			{
				ss >> temp;
				mKernel.width = temp;
				ss >> temp;
				mKernel.height = temp;
				ss >> temp;
				mKernel.stride = temp;
			}
			else if (word.substr(0, 2) == "NN")
			{
				myStructure.push_back(nnStruct());
				myStructure.back().layerType = word[2];
				ss >> temp;
				myStructure.back().width = temp;
				ss >> temp;
				myStructure.back().height = temp;
				ss >> temp;
				myStructure.back().depth = temp;
				myStructure.back().neuronNumber = myStructure.back().height*myStructure.back().width*myStructure.back().depth;
			}
			else if (word == "end")  break;
		}
	}
	else
	{
		cout << "Fisierul InputInfo.txt nu a putut fi deschis" << endl;
	}	
}

void dataFetcher::clearObj()
{
	this->imgVector.clear();
	this->image.release();
}

void dataFetcher::dataAugmentation()
{
	system("cls");
	cout << "Data Augm Menu:" << endl;
	while (1)
	{
		cout << endl << "Menu:"<<endl
			<< "[1] Adaugare Salt and Pepper" << endl
			<< "[2] Intoarce pe orizontala" << endl
			<< "[3] Gaussian Blur" << endl
			<< "[4] RGB Augmentation" << endl
			<< "[5] Rotatie, Translatie , Scalare" << endl
			<< "[0] Revenire Meniul Principal" << endl
			<< endl << "Select operation: ";

		unsigned operation;
		cin >> operation;

		if (operation == 0)
		{
			system("cls");
			break;
		}		
		else if (operation == 1)
		{ 			
			//Add Noise (Salt and Pepper)
			string filesPath, filesFile;
			unsigned filesNumber;

			cin.ignore();
			cout << "Cale folder poze: ";
			getline(cin, filesPath, '\n');
			if (filesPath.back() != '\\') filesPath += '\\';

			cout << "Cale fisier nume poze: ";
			getline(cin, filesFile, '\n');

			cout << "Numar fisiere: ";
			cin >> filesNumber;

			ifstream file;
			file.open(filesFile);

			if (file.is_open())
			{
				char q1;
				while (1)
				{
					cout << "Doriti mai multe modificari per poza?(Y/N): ";
					cin >> q1;
					if (q1 == 'Y' || q1 == 'N') break;
				}

				if (q1 == 'N')
				{
					string sCmd = "cd /D " + filesPath + " & mkdir SaltAndPepperSingle";
					char *cmd = new char[sCmd.length()+1];
					copy(sCmd.begin(), sCmd.end(), cmd);
					cmd[sCmd.length()] = '\0';
					system(cmd);

					unsigned min, max;
					cout << "Prag min: ";
					cin >> min;
					cout << "Prag max: ";
					cin >> max;

					for (unsigned nFile = 0; nFile < filesNumber && !file.eof(); nFile++)
					{
						string fileName;
						getline(file, fileName, '\n');
						if (fileName == "") break;
						cv::Mat image = getImage(fileName);
						image = dataAugm::_addNoise(image, min, max);
						cv::imwrite(filesPath + "SaltAndPepperSingle\\" + dataFetcher::formatName(fileName,"SP"), image);
					}
				}
				else
				{
					unsigned min, max, steps, delta;

					cout << "Prag min: ";
					cin >> min;
					cout << "Prag max: ";
					cin >> max;
					cout << "Pas: ";
					cin >> delta;
					cout << "Numar pasi: ";
					cin >> steps;

					string sCmd = "cd /D " + filesPath + " & mkdir SaltAndPepperBatch";
					char *cmd = new char[sCmd.length() + 1];
					copy(sCmd.begin(), sCmd.end(), cmd);
					cmd[sCmd.length()] = '\0';
					system(cmd);
					delete[] cmd;

					for (unsigned nFile = 0; nFile < filesNumber && !file.eof(); nFile++)
					{
						string fileName, folderName;
						getline(file, fileName, '\n');
						if (fileName == "") break;
						folderName = getFolderName(fileName);

						sCmd = "cd /D " + filesPath + "SaltAndPepperBatch & mkdir " + folderName;
						char *cmd = new char[sCmd.length() + 1];
						copy(sCmd.begin(), sCmd.end(), cmd);
						cmd[sCmd.length()] = '\0';
						system(cmd);
						delete[] cmd;

						for (unsigned nStep = 0; nStep < steps; nStep++)
						{
							cv::Mat image = getImage(fileName);
							image = dataAugm::_addNoise(image, min + nStep * delta, max - nStep * delta);
							cv::imwrite(filesPath + "SaltAndPepperBatch\\" + folderName + "\\" + dataFetcher::formatName(fileName,"SPB", nStep + 1), image);
						}
					}
				}
			}
			else
			{
				cout << "Fisierul nu a putut fi deschis!" << endl;
			}			
		}	
		else if (operation == 2)
		{
			//Flip Horizontal
			string filesPath, filesFile;
			unsigned filesNumber;

			cin.ignore();
			cout << "Cale folder: ";
			getline(cin, filesPath, '\n');
			if (filesPath.back() != '\\') filesPath += '\\';

			cout << "Cale fisier nume poze: ";
			getline(cin, filesFile, '\n');
			cout << "Numar fisiere: ";
			cin >> filesNumber;


			ifstream file;
			file.open(filesFile);
			if (file.is_open())
			{
				string sCmd = "cd /D " + filesPath + " & mkdir FlipHorizontal";
				char *cmd = new char[sCmd.length() + 1];
				copy(sCmd.begin(), sCmd.end(), cmd);
				cmd[sCmd.length()] = '\0';
				system(cmd);


				for (unsigned nFile = 0; nFile < filesNumber && !file.eof(); nFile++)
				{
					string fileName;
					getline(file, fileName, '\n');
					if (fileName == "") break;
					cv::Mat image = getImage(fileName);
					image = dataAugm::_flipHorizontal(image);
					cv::imwrite(filesPath + "FlipHorizontal\\" + dataFetcher::formatName(fileName,"FH"), image);
				}
			}
			else
			{
				cout << "Fisierul nu a putut fi deschis!" << endl
					<< "Nume fisier: " << filesFile << endl;
			}	
		}		
		else if (operation == 3)
		{
			//Gaussian Blur 
				
			string filesPath, filesFile;
			unsigned filesNumber;

			cin.ignore();
			cout << "Cale folder: ";
			getline(cin, filesPath, '\n');
			if (filesPath.back() != '\\') filesPath += '\\';
			cout << "Cale fisier nume poze: ";
			getline(cin, filesFile, '\n');
			cout << "Numar fisiere: ";
			cin >> filesNumber;

			ifstream file;
			file.open(filesFile);
			if (file.is_open())
			{
				char q1;
				while (1)
				{
					cout << "Doriti mai multe modificari per poza?(Y/N): ";
					cin >> q1;
					if (q1 == 'Y' || q1 == 'N') break;
				}
				if (q1 == 'N')
				{
					string sCmd = "cd /D " + filesPath + " & mkdir GaussianBlur";
					char *cmd = new char[sCmd.length() + 1];
					copy(sCmd.begin(), sCmd.end(), cmd);
					cmd[sCmd.length()] = '\0';
					system(cmd);

					unsigned val;
					cout << "Dimensiune Kernel: ";
					cin >> val;
					for (unsigned nFile = 0; nFile < filesNumber && !file.eof(); nFile++)
					{
						string fileName;
						getline(file, fileName, '\n');
						if (fileName == "") break;
						cv::Mat image = getImage(fileName);
						image = dataAugm::_GaussianBlur(image, val);
						cv::imwrite(filesPath + "GaussianBlur\\" + dataFetcher::formatName(fileName,"GB"), image);
					}
				}
				else
				{
					double val, delta;
					unsigned steps;

					cout << "Dimensiune initiala: ";
					cin >> val;
					cout << "Pas: ";
					cin >> delta;
					cout << "Numar pasi: ";
					cin >> steps;


					string sCmd = "cd /D " + filesPath + " & mkdir GaussianBlurBatch";
					char *cmd = new char[sCmd.length() + 1];
					copy(sCmd.begin(), sCmd.end(), cmd);
					cmd[sCmd.length()] = '\0';
					system(cmd);
					delete[] cmd;

					for (unsigned nFile = 0; nFile < filesNumber && !file.eof(); nFile++)
					{
						string fileName, folderName;
						getline(file, fileName, '\n');
						if (fileName == "") break;
						folderName = getFolderName(fileName);

						string sCmd = "cd /D " + filesPath + "GaussianBlurBatch & mkdir " + folderName;
						char *cmd = new char[sCmd.length() + 1];
						copy(sCmd.begin(), sCmd.end(), cmd);
						cmd[sCmd.length()] = '\0';
						system(cmd);
						delete[] cmd;

						for (unsigned nStep = 0; nStep < steps; nStep++)
						{
							cv::Mat image = getImage(fileName);
							image = dataAugm::_GaussianBlur(image, val + delta * nStep);
							cv::imwrite(filesPath + "GaussianBlurBatch\\" + folderName + "\\" + dataFetcher::formatName(fileName,"GB", nStep + 1), image);
						}
					}
				}
			}
			else
			{
				cout << "Fisierul nu a putut fi deschis!" << endl
					<< "Nume fisier: " << filesFile << endl;
			}
		}
		else if (operation == 4)
		{
			//RGB Augmentation

			string filesPath, filesFile;
			unsigned filesNumber;

			cin.ignore();
			cout << "Cale folder: ";
			getline(cin, filesPath, '\n');
			if (filesPath.back() != '\\') filesPath += '\\';
			cout << "Cale fisier nume poze: ";
			getline(cin, filesFile, '\n');
			cout << "Numar fisiere: ";
			cin >> filesNumber;

			ifstream file;
			file.open(filesFile);
			if (file.is_open())
			{
				char q1;
				while (1)
				{
					cout << "Doriti mai multe modificari per poza?(Y/N): ";
					cin >> q1;
					if (q1 == 'Y' || q1 == 'N') break;
				}
				if (q1 == 'N')
				{

					string sCmd = "cd /D " + filesPath + " & mkdir RGBAug";
					char *cmd = new char[sCmd.length() + 1];
					copy(sCmd.begin(), sCmd.end(), cmd);
					cmd[sCmd.length()] = '\0';
					system(cmd);

					double redP, greenP, blueP;
					cout << "Dati valoare  [R%]: ";
					cin >> redP;
					cout << "Dati valoare  [G%]: ";
					cin >> greenP;
					cout << "Dati valoare  [B%]: ";
					cin >> blueP;
					for (unsigned nFile = 0; nFile < filesNumber && !file.eof(); nFile++)
					{
						string fileName;
						getline(file, fileName, '\n');
						if (fileName == "") break;
						cv::Mat image = getImage(fileName);
						image = dataAugm::_RGB(image, redP, greenP, blueP);
						cv::imwrite(filesPath + "RGBAug\\" + dataFetcher::formatName(fileName,redP,greenP,blueP), image);
					}
				}
				else
				{
					double delta1, delta2, delta3;
					double redP, greenP, blueP;
					unsigned steps;

					cout << "Dati valoare  [R%]: ";
					cin >> redP;
					cout << "Dati valoare  [G%]: ";
					cin >> greenP;
					cout << "Dati valoare  [B%]: ";
					cin >> blueP;
					cout << "Pas [R]: ";
					cin >> delta1;
					cout << "Pas [G]: ";
					cin >> delta2;
					cout << "Pas [B]: ";
					cin >> delta3;
					cout << "Numar pasi: ";
					cin >> steps;

					string sCmd = "cd /D " + filesPath + " & mkdir RGBAugBatch";
					char *cmd = new char[sCmd.length() + 1];
					copy(sCmd.begin(), sCmd.end(), cmd);
					cmd[sCmd.length()] = '\0';
					system(cmd);
					delete[] cmd;

					for (unsigned nFile = 0; nFile < filesNumber && !file.eof(); nFile++)
					{
						string fileName, folderName;
						getline(file, fileName, '\n');
						if (fileName == "") break;
						folderName = getFolderName(fileName);

						string sCmd = "cd /D " + filesPath + "RGBAugBatch & mkdir " + folderName;
						char *cmd = new char[sCmd.length() + 1];
						copy(sCmd.begin(), sCmd.end(), cmd);
						cmd[sCmd.length()] = '\0';
						system(cmd);
						delete[] cmd;

						for (unsigned nStep = 0; nStep < steps; nStep++)
						{
							cv::Mat image = getImage(fileName);
							image = dataAugm::_RGB(image, redP + nStep * delta1, greenP + nStep * delta2, blueP + nStep * delta3);
							cv::imwrite(filesPath + "RGBAugBatch\\" + folderName + "\\" + dataFetcher::formatName(fileName, redP, greenP, blueP, nStep + 1), image);
						}
					}
				}
			}
			else
			{
				cout << "Fisierul nu a putut fi deschis!" << endl
					<< "Nume fisier: " << filesFile << endl;
			}
		}
		else if (operation == 5)
		{
			//Rotate, Translation, Scaling Augmentation
			//RGB Augmentation

			string filesPath, filesFile;
			unsigned filesNumber;

			cin.ignore();
			cout << "Cale folder: ";
			getline(cin, filesPath, '\n');
			if (filesPath.back() != '\\') filesPath += '\\';
			cout << "Cale fisier nume poze: ";
			getline(cin, filesFile, '\n');
			cout << "Numar fisiere: ";
			cin >> filesNumber;

			ifstream file;
			file.open(filesFile);
			if (file.is_open())
			{
				char q1;
				while (1)
				{
					cout << "Doriti mai multe modificari per poza?(Y/N): ";
					cin >> q1;
					if (q1 == 'Y' || q1 == 'N') break;
				}
				if (q1 == 'N')
				{
					string sCmd = "cd /D " + filesPath + " & mkdir RTSAug";
					char *cmd = new char[sCmd.length() + 1];
					copy(sCmd.begin(), sCmd.end(), cmd);
					cmd[sCmd.length()] = '\0';
					system(cmd);

					double traX, traY, rot, sca;
					cout << "Dati translatie[X - pixeli]: ";
					cin >> traX;
					cout << "Dati translatie[Y - pixeli]: ";
					cin >> traY;
					cout << "Dati rotatie: ";
					cin >> rot;
					cout << "Dati scalare: ";
					cin >> sca;
					for (unsigned nFile = 0; nFile < filesNumber && !file.eof(); nFile++)
					{
						string fileName;
						getline(file, fileName, '\n');
						if (fileName == "") break;
						cv::Mat image = getImage(fileName);
						image = dataAugm::_RTS(image, rot, traX, traY, sca);
						cv::imwrite(filesPath + "RTSAug\\" + dataFetcher::formatName(fileName, rot, traX, traY, sca), image);
					}
				}
				else
				{
					double delta1, delta2, delta3, delta4;
					double traX, traY, rot, sca;
					unsigned  steps;

					cout << "Dati translatie[X - pixeli]: ";
					cin >> traX;
					cout << "Dati translatie[Y - pixeli]: ";
					cin >> traY;
					cout << "Dati rotatie: ";
					cin >> rot;
					cout << "Dati scalare: ";
					cin >> sca;

					cout << "Pas [TX]: ";
					cin >> delta1;
					cout << "Pas [TY]: ";
					cin >> delta2;
					cout << "Pas [R]: ";
					cin >> delta3;
					cout << "Pas [S]: ";
					cin >> delta4;
					cout << "Numar pasi: ";
					cin >> steps;

					string sCmd = "cd /D " + filesPath + " & mkdir RTSAugBatch";
					char *cmd = new char[sCmd.length() + 1];
					copy(sCmd.begin(), sCmd.end(), cmd);
					cmd[sCmd.length()] = '\0';
					system(cmd);
					delete[] cmd;

					for (unsigned nFile = 0; nFile < filesNumber && !file.eof(); nFile++)
					{
						string fileName, folderName;
						getline(file, fileName, '\n');
						if (fileName == "") break;
						folderName = getFolderName(fileName);

						string sCmd = "cd /D " + filesPath + "RTSAugBatch & mkdir " + folderName;
						char *cmd = new char[sCmd.length() + 1];
						copy(sCmd.begin(), sCmd.end(), cmd);
						cmd[sCmd.length()] = '\0';
						system(cmd);
						delete[] cmd;

						for (unsigned nStep = 0; nStep < steps; nStep++)
						{
							cv::Mat image = getImage(fileName);
							image = dataAugm::_RTS(image, rot + nStep * delta3, traX + nStep * delta1, traY + nStep * delta2, sca + nStep * delta4);
							cv::imwrite(filesPath + "RTSAugBatch\\" + folderName + "\\" + dataFetcher::formatName(fileName, rot, traX, traY, sca, nStep + 1), image);
						}
					}
				}
			}
			else
			{
				cout << "Fisierul nu a putut fi deschis!" << endl
					<< "Nume fisier: " << filesFile << endl;
			}
		}
		else
		{
			cout << "Operatie invalida!" << endl;
		}
	}
}

cv::Mat dataFetcher::getImage(string path)
{
	cv::Mat imagine;
	imagine = cv::imread(path, cv::IMREAD_COLOR);
	return imagine;
}

string dataFetcher::formatName(string name, unsigned number)
{
	unsigned pos1, pos2;
	for (unsigned pName = name.length(); pName > 0; pName--)
	{
		if (name[pName] == '.')
		{
			pos2 = pName;
		}
		if (name[pName] == '\\')
		{
			pos1 = pName;
			break;
		}
	}
	name = name.substr(pos1 + 1, pos2 - pos1 - 1) + " (" + to_string(number) + ")" + name.substr(pos2, name.length() - pos2);
	return name;
}

string dataFetcher::formatName(string name, string tr)
{
	unsigned pos1, pos2;
	for (unsigned pName = name.length(); pName > 0; pName--)
	{
		if (name[pName] == '.')
		{
			pos2 = pName;
		}
		if (name[pName] == '\\')
		{
			pos1 = pName;
			break;
		}
	}
	name = name.substr(pos1 + 1, pos2 - pos1 - 1) + "_" + tr + name.substr(pos2, name.length() - pos2);
	return name;
}

string dataFetcher::formatName(string name, double R, double G, double B)
{
	unsigned pos1, pos2;
	for (unsigned pName = name.length(); pName > 0; pName--)
	{
		if (name[pName] == '.')
		{
			pos2 = pName;
		}
		if (name[pName] == '\\')
		{
			pos1 = pName;
			break;
		}
	}
	string cols;
	if (R != 0) cols += "R";
	if (G != 0)cols += "G";
	if (B != 0)cols += "B";
	name = name.substr(pos1 + 1, pos2 - pos1 - 1) + "_" + cols + name.substr(pos2, name.length() - pos2);
	return name;
}

string dataFetcher::formatName(string name, double R, double G, double B, unsigned number)
{
	unsigned pos1, pos2;
	for (unsigned pName = name.length(); pName > 0; pName--)
	{
		if (name[pName] == '.')
		{
			pos2 = pName;
		}
		if (name[pName] == '\\')
		{
			pos1 = pName;
			break;
		}
	}
	string cols;
	if (R != 0) cols += "R";
	if (G != 0)cols += "G";
	if (B != 0)cols += "B";
	name = name.substr(pos1 + 1, pos2 - pos1 - 1) + "_" + cols + " (" + to_string(number) + ")" + name.substr(pos2, name.length() - pos2);
	return name;
}

string dataFetcher::formatName(string name, double R, double Tx, double Ty, double S)
{
	unsigned pos1, pos2;
	for (unsigned pName = name.length(); pName > 0; pName--)
	{
		if (name[pName] == '.')
		{
			pos2 = pName;
		}
		if (name[pName] == '\\')
		{
			pos1 = pName;
			break;
		}
	}
	string ops;
	if (R != 0) ops += "Ro";
	if (Tx != 0)ops += "Tx";
	if (Ty != 0)ops += "Ty";
	if (S != 0)ops += "S";
	name = name.substr(pos1 + 1, pos2 - pos1 - 1) + "_" + ops + name.substr(pos2, name.length() - pos2);
	return name;
}

string dataFetcher::formatName(string name, double R, double Tx, double Ty, double S, unsigned number)
{
	unsigned pos1, pos2;
	for (unsigned pName = name.length(); pName > 0; pName--)
	{
		if (name[pName] == '.')
		{
			pos2 = pName;
		}
		if (name[pName] == '\\')
		{
			pos1 = pName;
			break;
		}
	}
	string ops;
	if (R != 0) ops += "Ro";
	if (Tx != 0)ops += "Tx";
	if (Ty != 0)ops += "Ty";
	if (S != 0)ops += "S";
	name = name.substr(pos1 + 1, pos2 - pos1 - 1) + "_" + ops + " (" + to_string(number) + ")" + name.substr(pos2, name.length() - pos2);
	return name;
}

string dataFetcher::formatName(string name, string tr, unsigned number)
{
	unsigned pos1, pos2;
	for (unsigned pName = name.length(); pName >0 ; pName--)
	{
		if (name[pName] == '.')
		{
			pos2 = pName;
		}
		if (name[pName] == '\\')
		{
			pos1 = pName;
			break;
		}
	}
	name = name.substr(pos1 + 1, pos2 - pos1 - 1) + "_" + tr + "_(" + to_string(number) + ")" + name.substr(pos2, name.length() - pos2);
	return name;
}

void dataFetcher::imagesSave(const string & path, const string &name,vector<cv::Mat> vectImg)
{
	for (unsigned vNum = 0; vNum < vectImg.size(); vNum++)
	{
		string fname = dataFetcher::formatName(name, vNum + 1);
		cv::imwrite(path + fname, vectImg[vNum]);
	}
}

void dataFetcher::resizeImage(const double & width, const double & height)
{
	double y = height / this->image.rows;
	double x = width / this->image.cols;
	cv::resize(this->image, this->image,cv::Size(), x, y, cv::INTER_LINEAR);
}

void dataFetcher::normalizeVector()
{	
	for (unsigned nPixel=0;nPixel<this->imgVector.size();nPixel++)
	{
		this->imgVector[nPixel] = this->imgVector[nPixel] / 256;
	}
}

void dataFetcher::setVector()
{
	vector<double> r,g,b;
	for (unsigned nRow = 0; nRow < this->image.rows; nRow++)
	{
		for (unsigned nCol = 0; nCol < this->image.cols; nCol++)
		{
			b.push_back(this->image.at<cv::Vec3b>(nRow, nCol).val[0]);
			g.push_back(this->image.at<cv::Vec3b>(nRow, nCol).val[1]);
			r.push_back(this->image.at<cv::Vec3b>(nRow, nCol).val[2]);
		}
	}
	this->imgVector.insert(this->imgVector.end(), r.begin(), r.end());
	this->imgVector.insert(this->imgVector.end(), g.begin(), g.end());
	this->imgVector.insert(this->imgVector.end(), b.begin(), b.end());
}

string dataFetcher::getFolderName(string fileName)
{
	unsigned posVal2 = fileName.length() - 1, posVal1;
	for (unsigned nPos = fileName.size() - 1; nPos > 0; nPos--)
	{
		if (fileName[nPos] == '.')
		{
			posVal2 = nPos;
			break;
		}
	}

	for (unsigned nPos = posVal2; nPos >= 0; nPos--)
	{
		if (fileName[nPos] == '\\')
		{
			posVal1 = nPos;
			break;
		}
	}
	return fileName.substr(posVal1 + 1, posVal2 - posVal1 - 1);
}
