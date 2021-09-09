#include "header.h"

char Real_Time_Testing() {
	system("Recording_Module.exe 3 input_file.wav input_file.txt");
	ifstream inPtr("input_file.txt");
	ReadIntensity(inPtr);
	ThrowInitialDisturbance();
	RepresentativeCepstrulCoefficients();
	ImportCodebook();
	GetObservationSequence();
	DisplayObservationSequence();
	char letter_digit;
	Prediction(letter_digit);
	return letter_digit;
}