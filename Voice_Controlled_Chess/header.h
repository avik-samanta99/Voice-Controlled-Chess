#include<SFML/Graphics.hpp>
#include<SFML/Audio.hpp>
#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include<fstream>
#include<string>
#include<vector>
#include<math.h>
using namespace std;
using namespace sf;

void ReadIntensity(ifstream&);
void ThrowInitialDisturbance();
void DCShift();
void Normalization();
long double STE(int);
int ZCR(int);
void StableFrames();
void MarkBoundary();
void SelectFrame(int);
void DisplayFrame();
void HammingWindow();
void AutoCorrelation(int);
void LevinsonDurbin(int);
void CepstrulCoefficients(int);
void RaisedSineWindow(int);
void RepresentativeCepstrulCoefficients();
void DisplayCoefficientsA();
void DisplayCoefficientsC();
void ImportCodebook();
void DisplayCodebook();
void TokhuraDistance(vector<long double>&, vector<long double>&, long double&);
void GetObservationSequence();
void DisplayObservationSequence();
void readA();
void readB();
void readPI();
void ReadInitialModelValues();
void InitializeHMMModel();
void InitializeFinalModel();
void ForwardProcedure();
void ScaledForwardProcedure();
void BackwardProcedure();
void ScaledBackwardProcedure();
void ViterbiAlgorithm();
void AlternateViterbiPreprocessing();
void AlternateViterbiAlgorithm();
void DisplayStateSequence();
void ComputeXi();
void ComputeGamma();
void UpdatePI();
void UpdateA();
void UpdateB();
void UpdateModelParameters();
void CopyFromTempModelValues();
void DisplayModel();
void Reestimation();
void ModifyFinalModel();
void AverageFinalModel();
void WriteFinalModelValues(string, string, string);
void TrainModel_digits();
void TrainModel_letters();
void readATest(string);
void readBTest(string);
void readPITest(string);
void InitializeModelValues(string);
void DigitPrediction(char&, long double&);
void LetterPrediction(char&, long double&);
void Prediction(char&);
void TestModel_digits();
void TestModel_letters();
void Train_Model();
void Test_Model();
char Real_Time_Testing();

class Position {
public:
    int x;
    int y;
    Position() : x(0), y(0) {}
    Position(int a, int b) : x(a), y(b) {}
};
bool whitePawnValidity(Position, Position, vector<vector<int>>&);
bool blackPawnValidity(Position, Position, vector<vector<int>>&);
bool RookValidity(Position, Position, vector<vector<int>>&);
bool BishopValidity(Position, Position, vector<vector<int>>&);
bool KnightValidity(Position, Position, vector<vector<int>>&);
bool QueenValidity(Position, Position, vector<vector<int>>&);
bool KingValidity(Position, Position, vector<vector<int>>&);

Position toPosition(char, char); 
std::string toMove(Position, Position);
void customizeMoveText(std::string);
void customizeTurnText(std::string);
void customizeCheckText(Text&);
void customizeButtonText(Text&);
void customizeWinnerText(std::string);
bool isValidMove(std::string);
void inCheck();
void loadPosition();
void makeMove(std::string);
void Play_Chess();
