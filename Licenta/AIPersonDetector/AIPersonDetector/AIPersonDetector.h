#pragma once
#include "neuralNetwork.h"
#include "dataFetcher.h"
#include "dataFlow.h"

void getTimeStamp(std::chrono::system_clock::time_point &newTimeStamp);
void coutTS(std::chrono::system_clock::time_point stop, std::chrono::system_clock::time_point start);
void showMenu();
void pathCorrector(string &path);