#include "AIPersonDetector.h"

void getTimeStamp(std::chrono::system_clock::time_point &newTimeStamp)
{
	newTimeStamp = chrono::system_clock::now();
}

void coutTS(std::chrono::system_clock::time_point stop, std::chrono::system_clock::time_point start)
{
	cout << "[Durata operatie: " << chrono::duration<double>(stop - start).count() << " secunde]" << endl;
}

void showMenu()
{
	cout
		<< "[1] Incarca Arhitectura Retea" << endl
		<< "[2] Initializare Arhitectura Retea" << endl
		<< "[3] Antrenare" << endl
		<< "[4] Rulare pe Imagini" << endl
		<< "[5] Rulare asistata" << endl
		<< "[6] Importa Ponderi" << endl
		<< "[7] Salveaza Ponderi" << endl
		<< "[8] Data Augmentation" << endl
		<< "[0] Exit" << endl
		<< "Selectati operatia: ";
}

void pathCorrector(string & path)
{
	for (unsigned nPos = path.length() - 1; nPos > 0; nPos--)
	{
		if (path[nPos - 1] == '/'  && path[nPos] == '/') path.insert(nPos + 1, 1, '/');
	}
	if (path[0] == '/') path.insert(path.begin(), '/');
	if (path.back() != '\\') path += '\\';
}