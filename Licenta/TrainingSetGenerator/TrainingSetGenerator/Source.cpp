#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

using namespace std;

string chars = "abcdefghijklmnopqrstuvwxyz1234567890";

string randNameGen(unsigned length)
{
	string result;
	for (unsigned chN = 0; chN < length; chN++)
	{
		result += chars[rand() % chars.length()];
	}
	return result;
}

string createDataSet(string filePath, string fileName,string fileType, string targetDir)
{
	if (targetDir.back() != '\\') targetDir += '\\';
	string newName = targetDir + randNameGen(15) + fileType;
	string sCmd = "copy \"" + filePath + "\" \"" + newName + "\"";
	char *cmd = new char[sCmd.length() + 1];
	copy(sCmd.begin(), sCmd.end(), cmd);
	cmd[sCmd.length()] = '\0';
	system(cmd);
	return newName;
}

void genInfo(string &targetVect, string &fileName, string &fileType, string &filePath, unsigned classesNum)
{
	string vec;
	size_t result = filePath.find("Class_");
	if (result == string::npos)
	{
		targetVect = string();
		fileName = string();
		fileType = string();
		filePath = string();		
	}
	else
	{
		size_t delim = filePath.find('\\', result);
		if (delim == string::npos || delim == result + 6) 
		{
			targetVect = string();
			fileName = string();
			fileType = string();
			filePath = string();
		}
		else
		{
			unsigned classNumber = stoi(filePath.substr(result + 6, delim - result - 5));
			for (unsigned cNum = 0; cNum < classesNum; cNum++)
			{
				if (cNum + 1 == classNumber) vec += " 1";
				else vec += " 0";
			}
			size_t delim1 = filePath.find('.', delim);
			targetVect = vec;
			fileName = filePath.substr(delim + 1, delim1 - delim-1);
			fileType = filePath.substr(delim1 , filePath.length() - delim1);
		}
	}
}

int main()
{
	srand(time(NULL));
	string imagesList,
		targetFilePath,
		targetFolderPath;
	unsigned classNum;
	ifstream inpFile;
	ofstream outFile;
	cout << "Cale fisier lista imagini: ";
	getline(cin, imagesList, '\n');
	cout << "Cale fisier iesire: ";
	getline(cin, targetFilePath, '\n');
	cout << "Cale folder dataset nou: ";
	getline(cin, targetFolderPath, '\n');
	cout << "Numar clase: ";
	cin >> classNum;
	inpFile.open(imagesList);
	if (inpFile.is_open())
	{
		outFile.open(targetFilePath, ios::trunc);
		while (!inpFile.eof())
		{
			string file, vect, fileName, fileType;
			getline(inpFile, file, '\n');
			genInfo(vect, fileName, fileType, file, classNum);
			if (vect.length() != 0)
			{
				string newName = createDataSet(file, fileName, fileType, targetFolderPath);
				outFile << newName << ";" << vect << endl;
			}
		}		
	}
	else
	{
		cout << "Fisierul nu a putut fi deschis!" << endl;
	}
	system("pause");
	return 0;
}