#include "header.h"
using namespace std;

long double pi = 3.1428571428;
int recordingTime = 3;
int frequency = 16000;
int inputSampleSize = recordingTime * frequency;
int frameSize = 320;
int slide = (frameSize / 4);
int framesPerSecond = frequency / frameSize;
int totalFrames = recordingTime * framesPerSecond;
int framesOfDisturbance = 50;
int DCShiftFrame = 5;
long double standardMaxIntensity = 10000.0;
int LPCOrder = 12;
int stableFrame = 60; // to change
int codebookSize = 32;
int codeVectorSize = LPCOrder;
int states = 5;
int trainingSamples = 30;
int testingSamples = 10;
long double threshold = 1e-300;
long double tokhuraWeights[] = { 1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0 };

vector<long double> inputSampleIntensity;
vector<long double> tempFrame(frameSize);
pair<int, int> boundary;
vector<vector<long double>> CorrelationR(stableFrame, vector<long double>(LPCOrder + 1));
vector<vector<long double>> CoefficientsA(stableFrame, vector<long double>(LPCOrder + 1));
vector<vector<long double>> CoefficientsC(stableFrame, vector<long double>(LPCOrder + 1));
vector<vector<long double>> Codebook(codebookSize, vector<long double>(codeVectorSize));

vector<vector<long double>> alpha(stableFrame, vector<long double>(states));
vector<vector<long double>> beta(stableFrame, vector<long double>(states));
vector<long double> c_t(stableFrame);
vector<vector<long double>> alphaLocal(stableFrame, vector<long double>(states));
vector<vector<long double>> betaLocal(stableFrame, vector<long double>(states));
vector<vector<long double>> delta(stableFrame, vector<long double>(states));
vector<vector<int>> psi(stableFrame, vector<int>(states));
vector<vector<vector<long double>>> xi(stableFrame, vector<vector<long double>>(states, vector<long double>(states)));
vector<vector<long double>> gamma(stableFrame, vector<long double>(states));

vector<long double> initialPI(states);
vector<vector<long double>> initialA(states, vector<long double>(states));
vector<vector<long double>> initialB(states, vector<long double>(codebookSize));
vector<long double> modelPI(states);
vector<vector<long double>> modelA(states, vector<long double>(states));
vector<vector<long double>> modelB(states, vector<long double>(codebookSize));
vector<long double> logModelPI(states);
vector<vector<long double>> logModelA(states, vector<long double>(states));
vector<vector<long double>> logModelB(states, vector<long double>(codebookSize));
vector<long double> tempPI(states);
vector<vector<long double>> tempA(states, vector<long double>(states));
vector<vector<long double>> tempB(states, vector<long double>(codebookSize));
vector<long double> finalPI(states);
vector<vector<long double>> finalA(states, vector<long double>(states));
vector<vector<long double>> finalB(states, vector<long double>(codebookSize));

long double probability;
vector<int> ObservationSequence(stableFrame);
vector<int> StateSequence(stableFrame);
long double P_star, prevP_star;


void ReadIntensity(ifstream& inPtr) {
	string str;
	for (int j = 0; j < 5; j++)
		getline(inPtr, str);
	long double curIntensity;
	inPtr >> curIntensity;
	while (inPtr) {
		inputSampleIntensity.push_back(curIntensity);
		inPtr >> curIntensity;
	}
}

void ThrowInitialDisturbance() {
	inputSampleIntensity.erase(inputSampleIntensity.begin(), inputSampleIntensity.begin() + (frameSize * framesOfDisturbance));
	inputSampleSize -= (frameSize * framesOfDisturbance);
}

void DCShift() {
	long double totalIntensity = 0.0, avgIntensity;
	for (int i = 0; i < (DCShiftFrame * frameSize); i++)
		totalIntensity += inputSampleIntensity[i];
	avgIntensity = totalIntensity / (long double)((long double)DCShiftFrame * (long double)frameSize);
	for (int i = 0; i < (int)inputSampleIntensity.size(); i++)
		inputSampleIntensity[i] -= avgIntensity;
}

void Normalization() {
	long double maxIntensity = 0.0;
	for (int i = 0; i < (int)inputSampleIntensity.size(); i++)
		if (abs(inputSampleIntensity[i]) > maxIntensity)
			maxIntensity = inputSampleIntensity[i];
	long double normalizationFactor = standardMaxIntensity / maxIntensity;
	for (int i = 0; i < (int)inputSampleIntensity.size(); i++)
		inputSampleIntensity[i] *= normalizationFactor;
}

long double STE(int frameNo) {
	long double ste = 0.0;
	int start = (frameNo * slide);
	for (int i = start; i < (start + frameSize); i++)
		ste += (inputSampleIntensity[i] * inputSampleIntensity[i]);
	ste /= frameSize;
	return ste;
}

int ZCR(int frameNo) {
	int zcr = 0;
	int start = (frameNo * slide);
	for (int i = start + 1; i < (start + frameSize); i++)
		if (inputSampleIntensity[i - 1] * inputSampleIntensity[i] <= 0.0) zcr++;
	return zcr;
}

void StableFrames() {
	/*
	int frames = inputSampleIntensity.size() / slide - 1;
	inputSampleIntensity.erase(inputSampleIntensity.begin() + (frames * slide + (slide / 3)), inputSampleIntensity.end());
	int maxEnergyFrame;
	long double maxEnergy = 0.0;
	for(int i = 0; i < frames; i ++){
		long double ste = STE(i);
		if(ste > maxEnergy){
			maxEnergy = ste;
			maxEnergyFrame = i;
		}
	}
	inputSampleIntensity.erase(inputSampleIntensity.begin(), inputSampleIntensity.begin() + ((maxEnergyFrame - (stableFrame / 2)) * slide));
	inputSampleIntensity.erase(inputSampleIntensity.begin() + ((stableFrame * slide) + (slide / 3)), inputSampleIntensity.end());
	*/
	inputSampleIntensity.erase(inputSampleIntensity.begin(), inputSampleIntensity.begin() + (boundary.first * slide));
	inputSampleIntensity.erase(inputSampleIntensity.begin() + ((stableFrame * slide) + (slide * 3)), inputSampleIntensity.end());
}

void MarkBoundary() {
	int frames = ((int)inputSampleIntensity.size() - (slide * 3)) / slide;
	vector<long double> cumulativeSTE(frames);
	long double maxSTE = 0.0;
	for (int i = 0; i < frames; i++) {
		if (i == 0) cumulativeSTE[i] = STE(i);
		else cumulativeSTE[i] = STE(i) + cumulativeSTE[i - 1];
		if (i >= stableFrame && cumulativeSTE[i] - cumulativeSTE[i - stableFrame + 1] > maxSTE) {
			maxSTE = cumulativeSTE[i] - cumulativeSTE[i - stableFrame + 1];
			boundary.first = i - stableFrame + 1;
			boundary.second = i;
		}
	}
}

void SelectFrame(int frameNo) {
	int start = frameNo * slide;
	for (int i = 0; i < frameSize; i++)
		tempFrame[i] = inputSampleIntensity[start + i];
}

void DisplayFrame() {
	for (int i = 0; i < frameSize; i++) {
		printf("%0.5lf ", tempFrame[i]);
	} cout << endl;
}

void HammingWindow() {
	for (int j = 0; j < frameSize; j++) {
		long double weight = 0.54 - 0.46 * cos(2 * pi * (long double)j / (long double)((long double)frameSize - 1.0));
		tempFrame[j] *= weight;
	}
}

void AutoCorrelation(int frameNo) {
	for (int j = 0; j <= LPCOrder; j++) {
		CorrelationR[frameNo][j] = 0.0;
		for (int k = 0; k < (frameSize - j); k++)
			CorrelationR[frameNo][j] += tempFrame[k] * tempFrame[k + j];
	}
}

void LevinsonDurbin(int frameNo) {
	vector<vector<long double>> alphaAi(LPCOrder + 1, vector<long double>(LPCOrder + 1));
	vector<long double> residualError(LPCOrder + 1);
	residualError[0] = CorrelationR[frameNo][0];
	for (int j = 1; j <= LPCOrder; j++) {
		long double subtract = 0.0;
		for (int k = 1; k < j; k++)
			subtract += alphaAi[j - 1][k] * CorrelationR[frameNo][j - k];
		alphaAi[j][j] = (CorrelationR[frameNo][j] - subtract) / residualError[j - 1];
		for (int k = 1; k < j; k++)
			alphaAi[j][k] = alphaAi[j - 1][k] - alphaAi[j][j] * alphaAi[j - 1][j - k];
		residualError[j] = (1 - alphaAi[j][j] * alphaAi[j][j]) * residualError[j - 1];
	}
	CoefficientsA[frameNo][0] = 0.0;
	for (int j = 1; j <= LPCOrder; j++)
		CoefficientsA[frameNo][j] = alphaAi[LPCOrder][j];
}

void CepstrulCoefficients(int frameNo) {
	CoefficientsC[frameNo][0] = log(CorrelationR[frameNo][0] * CorrelationR[frameNo][0]) / log((long double)2.0);
	for (int i = 1; i <= LPCOrder; i++) {
		CoefficientsC[frameNo][i] = CoefficientsA[frameNo][i];
		for (int j = 1; j < i; j++) {
			CoefficientsC[frameNo][i] += ((long double)j * CoefficientsC[frameNo][j] * CoefficientsA[frameNo][i - j] / (long double)i);
		}
	}
}

void RaisedSineWindow(int frameNo) {
	for (int i = 1; i <= LPCOrder; i++) {
		long double w = 1 + (long double)LPCOrder * sin(pi * (long double)i / (long double)LPCOrder) / 2.0;
		CoefficientsC[frameNo][i] *= w;
	}
}

void RepresentativeCepstrulCoefficients() {
	// ThrowInitialDisturbance();
	DCShift();
	Normalization();
	MarkBoundary();
	StableFrames();
	for (int j = 0; j < stableFrame; j++) {
		SelectFrame(j);
		HammingWindow();
		AutoCorrelation(j);
		LevinsonDurbin(j);
		CepstrulCoefficients(j);
		RaisedSineWindow(j);
	}
}

void DisplayCoefficientsA() {
	for (int i = 0; i < stableFrame; i++) {
		cout << i + 1 << ")  ";
		for (int j = 1; j <= LPCOrder; j++) {
			printf("%0.5lf ", CoefficientsA[i][j]);
		}
		cout << endl;
	}
}

void DisplayCoefficientsC() {
	for (int i = 0; i < stableFrame; i++) {
		cout << i + 1 << ")  ";
		for (int j = 1; j <= LPCOrder; j++) {
			printf("%0.5lf ", CoefficientsC[i][j]);
		}
		cout << endl;
	}
}

void ImportCodebook() {
	ifstream inPtr("Precomputed/Codebook.txt");
	long double curValue;
	inPtr >> curValue;
	for (int i = 0; i < codebookSize; i++) {
		for (int j = 0; j < codeVectorSize; j++) {
			Codebook[i][j] = curValue;
			inPtr >> curValue;
		}
	}
}

void DisplayCodebook() {
	for (int i = 0; i < codebookSize; i++) {
		cout << i + 1 << ")  ";
		for (int j = 0; j < LPCOrder; j++) {
			printf("%0.5lf ", Codebook[i][j]);
		}
		cout << endl;
	}
}

void TokhuraDistance(vector<long double>& refVector, vector<long double>& curVector, long double& distance) {
	distance = 0.0;
	for (int j = 1; j <= LPCOrder; j++) {
		distance += tokhuraWeights[j - 1] * (refVector[j - 1] - curVector[j]) * (refVector[j - 1] - curVector[j]);
	}
}

void GetObservationSequence() {
	for (int i = 0; i < stableFrame; i++) {
		long double minDist = 1e15, distance;
		int minDistIndex = -1;
		for (int j = 0; j < codebookSize; j++) {
			TokhuraDistance(Codebook[j], CoefficientsC[i], distance);
			if (distance < minDist) {
				minDist = distance;
				minDistIndex = j;
			}
		}
		ObservationSequence[i] = minDistIndex;
	}
}

void DisplayObservationSequence() {
	for (int i = 0; i < stableFrame; i++)
		cout << ObservationSequence[i] << " ";
	cout << endl;
}

void readA() {
	ifstream inPtr;
	string directory = "Precomputed/A_MATRIX.txt";
	inPtr.open(directory);
	for (int i = 0; i < states; i++) {
		for (int j = 0; j < states; j++) {
			inPtr >> initialA[i][j];
		}
	}
}

void readB() {
	ifstream inPtr;
	inPtr.open("Precomputed/B_MATRIX.txt");
	for (int i = 0; i < states; i++) {
		for (int j = 0; j < codebookSize; j++) {
			inPtr >> initialB[i][j];
		}
	}
}

void readPI() {
	ifstream inPtr;
	inPtr.open("Precomputed/PI_MATRIX.txt");
	for (int j = 0; j < states; j++) {
		inPtr >> initialPI[j];
	}
}

void ReadInitialModelValues() {
	readA();
	readB();
	readPI();
}

void InitializeHMMModel() {
	for (int i = 0; i < states; i++) {
		modelPI[i] = initialPI[i];
		for (int j = 0; j < states; j++)
			modelA[i][j] = initialA[i][j];
		for (int j = 0; j < codebookSize; j++)
			modelB[i][j] = initialB[i][j];
	}
}

void InitializeFinalModel() {
	for (int i = 0; i < states; i++) {
		finalPI[i] = 0.0;
		for (int j = 0; j < states; j++)
			finalA[i][j] = 0.0;
		for (int j = 0; j < codebookSize; j++)
			finalB[i][j] = 0.0;
	}
}

void ForwardProcedure() {
	long double product;
	probability = 0.0;
	for (int i = 0; i < states; i++)
		alpha[0][i] = modelPI[i] * modelB[i][ObservationSequence[0]];
	for (int t = 0; t < stableFrame - 1; t++) {
		for (int j = 0; j < states; j++) {
			product = 0.0;
			for (int i = 0; i < states; i++) {
				product += alpha[t][i] * modelA[i][j];
			}
			alpha[t + 1][j] = modelB[j][ObservationSequence[t + 1]] * product;
		}
	}
	for (int i = 0; i < states; i++)
		probability += alpha[stableFrame - 1][i];
}

void ScaledForwardProcedure() {
	long double product;
	probability = c_t[0] = 0.0;
	for (int i = 0; i < states; i++) {
		alphaLocal[0][i] = modelPI[i] * modelB[i][ObservationSequence[0]];
		c_t[0] += alphaLocal[0][i];
	}
	for (int i = 0; i < states; i++)
		alpha[0][i] = alphaLocal[0][i] / c_t[0];
	for (int t = 0; t < stableFrame - 1; t++) {
		c_t[t + 1] = 0.0;
		for (int j = 0; j < states; j++) {
			product = 0.0;
			for (int i = 0; i < states; i++) {
				product += alpha[t][i] * modelA[i][j];
			}
			alphaLocal[t + 1][j] = modelB[j][ObservationSequence[t + 1]] * product;
			c_t[t + 1] += alphaLocal[t + 1][j];
		}
		for (int j = 0; j < states; j++) {
			alpha[t + 1][j] = alphaLocal[t + 1][j] / c_t[t + 1];
		}
	}
	for (int i = 0; i < states; i++)
		probability += alpha[stableFrame - 1][i];
}

void BackwardProcedure() {
	for (int i = 0; i < states; i++)
		beta[stableFrame - 1][i] = 1.0;
	for (int t = stableFrame - 2; t >= 0; t--) {
		for (int i = 0; i < states; i++) {
			beta[t][i] = 0.0;
			for (int j = 0; j < states; j++) {
				beta[t][i] += modelA[i][j] * modelB[j][ObservationSequence[t + 1]] * beta[t + 1][j];
			}
		}
	}
}

void ScaledBackwardProcedure() {
	for (int i = 0; i < states; i++)
		beta[stableFrame - 1][i] = 1.0 / c_t[stableFrame - 1];
	for (int t = stableFrame - 2; t >= 0; t--) {
		for (int i = 0; i < states; i++) {
			beta[t][i] = 0.0;
			for (int j = 0; j < states; j++) {
				beta[t][i] += modelA[i][j] * modelB[j][ObservationSequence[t + 1]] * beta[t + 1][j];
			}
			beta[t][i] /= c_t[t];
		}
	}
}

void ViterbiAlgorithm() {
	for (int i = 0; i < states; i++) {
		delta[0][i] = modelPI[i] * modelB[i][ObservationSequence[0]];
		psi[0][i] = -1;
	}
	long double maximum;
	for (int t = 1; t < stableFrame; t++) {
		for (int i = 0; i < states; i++) {
			maximum = 0.0;
			for (int j = 0; j < states; j++) {
				if (delta[t - 1][j] * modelA[j][i] > maximum) {
					maximum = delta[t - 1][j] * modelA[j][i];
					psi[t][i] = j;
				}
			}
			delta[t][i] = maximum * modelB[i][ObservationSequence[t]];
		}
	}
	P_star = 0.0;
	for (int i = 0; i < states; i++) {
		if (delta[stableFrame - 1][i] > P_star) {
			P_star = delta[stableFrame - 1][i];
			StateSequence[stableFrame - 1] = i;
		}
	}
	for (int t = stableFrame - 2; t >= 0; t--) {
		StateSequence[t] = psi[t + 1][StateSequence[t + 1]];
	}
}

void AlternateViterbiPreprocessing() {
	for (int i = 0; i < states; i++) {
		logModelPI[i] = log(modelPI[i]);
		for (int j = 0; j < states; j++)
			logModelA[i][j] = log(modelA[i][j]);
		for (int j = 0; j < codebookSize; j++)
			logModelB[i][j] = log(modelB[i][j]);
	}
}

void AlternateViterbiAlgorithm() {
	AlternateViterbiPreprocessing();
	for (int i = 0; i < states; i++) {
		delta[0][i] = logModelPI[i] + logModelB[i][ObservationSequence[0]];
		psi[0][i] = -1;
	}
	long double maximum;
	for (int t = 1; t < stableFrame; t++) {
		for (int i = 0; i < states; i++) {
			maximum = -10000.0;
			for (int j = 0; j < states; j++) {
				if (delta[t - 1][j] + logModelA[j][i] > maximum) {
					maximum = delta[t - 1][j] + logModelA[j][i];
					psi[t][i] = j;
				}
			}
			delta[t][i] = maximum + logModelB[i][ObservationSequence[t]];
		}
	}
	P_star = -10000.0;
	for (int i = 0; i < states; i++) {
		if (delta[stableFrame - 1][i] > P_star) {
			P_star = delta[stableFrame - 1][i];
			StateSequence[stableFrame - 1] = i;
		}
	}
	// cout << P_star << " ";
	for (int t = stableFrame - 2; t >= 0; t--) {
		StateSequence[t] = psi[t + 1][StateSequence[t + 1]];
	}
}

void DisplayStateSequence() {
	for (int i = 0; i < stableFrame; i++)
		cout << StateSequence[i] << " ";
	cout << endl;
}

void ComputeXi() {
	long double denominator;
	for (int t = 0; t < stableFrame - 1; t++) {
		denominator = 0.0;
		for (int i = 0; i < states; i++)
			for (int j = 0; j < states; j++)
				denominator += (alpha[t][i] * modelA[i][j] * modelB[j][ObservationSequence[t + 1]] * beta[t + 1][j]);
		for (int i = 0; i < states; i++)
			for (int j = 0; j < states; j++)
				xi[t][i][j] = (alpha[t][i] * modelA[i][j] * modelB[j][ObservationSequence[t + 1]] * beta[t + 1][j]) / denominator;
	}
}

void ComputeGamma() {
	long double denominator;
	for (int t = 0; t < stableFrame; t++) {
		denominator = 0.0;
		for (int i = 0; i < states; i++)
			denominator += (alpha[t][i] * beta[t][i]);
		for (int i = 0; i < states; i++)
			gamma[t][i] = (alpha[t][i] * beta[t][i]) / denominator;
	}
}

void UpdatePI() {
	for (int i = 0; i < states; i++) {
		tempPI[i] = gamma[0][i];
	}
}

void UpdateA() {
	long double numerator, denominator;
	for (int i = 0; i < states; i++) {
		denominator = 0.0;
		for (int t = 0; t < stableFrame - 1; t++)
			denominator += gamma[t][i];
		for (int j = 0; j < states; j++) {
			numerator = 0.0;
			for (int t = 0; t < stableFrame - 1; t++)
				numerator += xi[t][i][j];
			tempA[i][j] = numerator / denominator;
		}
	}
}

void UpdateB() {
	long double numerator, denominator;
	for (int j = 0; j < states; j++) {
		for (int k = 0; k < codebookSize; k++) {
			numerator = 0.0;
			denominator = 0.0;
			for (int t = 0; t < stableFrame; t++) {
				denominator += gamma[t][j];
				if (ObservationSequence[t] == k)
					numerator += gamma[t][j];
			}
			tempB[j][k] = max((long double)1e-30, numerator / denominator);
		}
	}
}

void UpdateModelParameters() {
	ComputeXi();
	ComputeGamma();
	UpdatePI();
	UpdateA();
	UpdateB();
}

void CopyFromTempModelValues() {
	for (int i = 0; i < states; i++) {
		modelPI[i] = tempPI[i];
		for (int j = 0; j < states; j++)
			modelA[i][j] = tempA[i][j];
		for (int j = 0; j < codebookSize; j++)
			modelB[i][j] = tempB[i][j];
	}
}

void DisplayModel() {
	cout << "Matrix A :-\n";
	for (int i = 0; i < states; i++) {
		for (int j = 0; j < states; j++)
			cout << modelA[i][j] << " ";
		cout << endl;
	}
	cout << "Matrix B :-\n";
	for (int i = 0; i < states; i++) {
		for (int j = 0; j < codebookSize; j++)
			cout << modelB[i][j] << " ";
		cout << endl;
	}
	cout << "Matrix PI :-\n";
	for (int i = 0; i < states; i++) {
		printf("%0.5lf ", modelPI[i]);
	} cout << endl << endl;
}

void Reestimation() {
	ReadInitialModelValues();
	InitializeHMMModel();
	// ViterbiAlgorithm();
	AlternateViterbiAlgorithm();
	// DisplayStateSequence();
	int count = 0;
	do {
		// DisplayModel();
		// prevP_star = P_star;
		ScaledForwardProcedure();
		ScaledBackwardProcedure();
		UpdateModelParameters();
		CopyFromTempModelValues();
		// ViterbiAlgorithm();
		AlternateViterbiAlgorithm();
		// cout << P_star << endl;
		// DisplayStateSequence();
		count++;
	} while (count < 40/* && abs(prevP_star - P_star) < threshold*/);
	// DisplayModel();
	cout << "Final P* for the training file : " << P_star << endl;
}

void ModifyFinalModel() {
	for (int i = 0; i < states; i++) {
		finalPI[i] += modelPI[i];
		for (int j = 0; j < states; j++)
			finalA[i][j] += modelA[i][j];
		for (int j = 0; j < codebookSize; j++)
			finalB[i][j] += modelB[i][j];
	}
}

void AverageFinalModel() {
	for (int i = 0; i < states; i++) {
		finalPI[i] /= (long double)trainingSamples;
		for (int j = 0; j < states; j++)
			finalA[i][j] /= (long double)trainingSamples;
		for (int j = 0; j < codebookSize; j++)
			finalB[i][j] /= (long double)trainingSamples;
	}
}

void WriteFinalModelValues(string directoryA, string directoryB, string directoryPI) {
	ofstream outPtrA(directoryA), outPtrB(directoryB), outPtrPI(directoryPI);
	for (int i = 0; i < states; i++) {
		for (int j = 0; j < states; j++)
			outPtrA << finalA[i][j] << " ";
		outPtrA << endl;
	}
	for (int i = 0; i < states; i++) {
		for (int j = 0; j < codebookSize; j++)
			outPtrB << finalB[i][j] << " ";
		outPtrB << endl;
	}
	for (int i = 0; i < states; i++) {
		outPtrPI << finalPI[i] << " ";
	} outPtrPI << endl;
}

void TrainModel_digits() {
	string outA = "Models/204101016_x_A.txt", outB = "Models/204101016_x_B.txt", outPI = "Models/204101016_x_PI.txt", extension = ".txt";
	ImportCodebook();
	for (char c = '1'; c <= '8'; c++) {
		cout << "\nTraining model for digit : " << c << " ...\n";
		string inputDirectory = "Data Files/204101016_x_";
		inputDirectory[21] = c;
		outA[17] = outB[17] = outPI[17] = c;
		InitializeFinalModel();
		// cout << "P* values for digit : " << c << " :-\n";
		for (int j = 1; j <= trainingSamples; j++) {
			inputDirectory.append(to_string((long long)j));
			inputDirectory.append(extension);
			ifstream inPtr(inputDirectory);
			ReadIntensity(inPtr);
			RepresentativeCepstrulCoefficients();
			// DisplayCoefficientsC();
			GetObservationSequence();
			// DisplayObservationSequence();
			Reestimation();
			// cout << endl;
			// DisplayModel();
			ModifyFinalModel();
			inputSampleIntensity.clear();
			inputDirectory.erase(inputDirectory.begin() + 23, inputDirectory.end());
			// cout << "done\n";
		}
		AverageFinalModel();
		WriteFinalModelValues(outA, outB, outPI);
		// cout << "done" << i << endl;
	}
}

void TrainModel_letters() {
	string outA = "Models/204101016_x_A.txt", outB = "Models/204101016_x_B.txt", outPI = "Models/204101016_x_PI.txt", extension = ".txt";
	ImportCodebook();
	for (char c = 'a'; c <= 'h'; c++) {
		cout << "\nTraining model for letter : " << c << " ...\n";
		string inputDirectory = "Data Files/204101016_x_";
		inputDirectory[21] = c;
		outA[17] = outB[17] = outPI[17] = c;
		InitializeFinalModel();
		// cout << "P* values for letter : " << c << " :-\n";
		for (int j = 1; j <= trainingSamples; j++) {
			inputDirectory.append(to_string((long long)j));
			inputDirectory.append(extension);
			ifstream inPtr(inputDirectory);
			ReadIntensity(inPtr);
			RepresentativeCepstrulCoefficients();
			// DisplayCoefficientsC();
			GetObservationSequence();
			// DisplayObservationSequence();
			Reestimation();
			// cout << endl;
			// DisplayModel();
			ModifyFinalModel();
			inputSampleIntensity.clear();
			inputDirectory.erase(inputDirectory.begin() + 23, inputDirectory.end());
			// cout << "done\n";
		}
		AverageFinalModel();
		WriteFinalModelValues(outA, outB, outPI);
		// cout << "done" << i << endl;
	}
}

void readATest(string inDir) {
	inDir.append("A");
	inDir.append(".txt");
	ifstream inPtr(inDir);
	for (int i = 0; i < states; i++) {
		for (int j = 0; j < states; j++) {
			inPtr >> modelA[i][j];
		}
	}
}

void readBTest(string inDir) {
	inDir.append("B");
	inDir.append(".txt");
	ifstream inPtr(inDir);
	for (int i = 0; i < states; i++) {
		for (int j = 0; j < codebookSize; j++) {
			inPtr >> modelB[i][j];
		}
	}
}

void readPITest(string inDir) {
	inDir.append("PI");
	inDir.append(".txt");
	ifstream inPtr(inDir);
	for (int i = 0; i < states; i++) {
		inPtr >> modelPI[i];
	}
}

void InitializeModelValues(string inDir) {
	readATest(inDir);
	readBTest(inDir);
	readPITest(inDir);
}

void DigitPrediction(char& predictedDigit, long double& maxProbability) {
	string modelDirectory = "Models/204101016_x_";
	for (char c = '1'; c <= '8'; c++) {
		modelDirectory[17] = c;
		InitializeModelValues(modelDirectory);
		// DisplayModel();
		ForwardProcedure();
		cout << "The probability for the model w.r.t. digit " << c << " : " << probability << endl;
		if (probability > maxProbability) {
			maxProbability = probability;
			predictedDigit = c;
		}
	}
}

void LetterPrediction(char& predictedLetter, long double& maxProbability) {
	string modelDirectory = "Models/204101016_x_";
	for (char c = 'a'; c <= 'h'; c++) {
		modelDirectory[17] = c;
		InitializeModelValues(modelDirectory);
		// DisplayModel();
		ForwardProcedure();
		cout << "The probability for the model w.r.t. letter " << c << " : " << probability << endl;
		if (probability > maxProbability) {
			maxProbability = probability;
			predictedLetter = c;
		}
	}
}

void Prediction(char& item) {
	long double maxProbability_digit = 0.0;
	long double maxProbability_letter = 0.0;
	char digit, letter;
	DigitPrediction(digit, maxProbability_digit);
	LetterPrediction(letter, maxProbability_letter);
	if (maxProbability_digit >= maxProbability_letter) item = digit;
	else item = letter;
}

void TestModel_digits() {
	string extension = ".txt";
	int correctPrediction = 0;
	char digit;
	ImportCodebook();
	for (char c = '1'; c <= '8'; c++) {
		string inputDirectory = "Data Files/204101016_x_";
		inputDirectory[21] = c;
		for (int j = 31; j < 31 + testingSamples; j++) {
			inputDirectory.append(to_string((long long)j));
			inputDirectory.append(extension);
			cout << "Testing for the file : " << inputDirectory << endl;
			ifstream inPtr(inputDirectory);
			ReadIntensity(inPtr);
			RepresentativeCepstrulCoefficients();
			// DisplayCoefficientsC();
			GetObservationSequence();
			// DisplayObservationSequence();
			Prediction(digit);
			cout << "Actual digit : " << c << ", Predicted digit : " << digit << ", Prediction : ";
			if (digit == c) cout << "Right\n\n", correctPrediction++;
			else cout << "Wrong\n\n";
			inputSampleIntensity.clear();
			inputDirectory.erase(inputDirectory.begin() + 23, inputDirectory.end());
		}
	}
	cout << "Prediction Accuracy : " << (long double)correctPrediction / (long double)80 << endl;
}

void TestModel_letters() {
	string extension = ".txt";
	int correctPrediction = 0;
	char letter;
	ImportCodebook();
	for (char c = 'a'; c <= 'h'; c++) {
		string inputDirectory = "Data Files/204101016_x_";
		inputDirectory[21] = c;
		for (int j = 31; j < 31 + testingSamples; j++) {
			inputDirectory.append(to_string((long long)j));
			inputDirectory.append(extension);
			cout << "Testing for the file : " << inputDirectory << endl;
			ifstream inPtr(inputDirectory);
			ReadIntensity(inPtr);
			RepresentativeCepstrulCoefficients();
			// DisplayCoefficientsC();
			GetObservationSequence();
			// DisplayObservationSequence();
			Prediction(letter);
			cout << "Actual letter : " << c << ", Predicted letter : " << letter << ", Prediction : ";
			if (letter == c) cout << "Right\n\n", correctPrediction++;
			else cout << "Wrong\n\n";
			inputSampleIntensity.clear();
			inputDirectory.erase(inputDirectory.begin() + 23, inputDirectory.end());
		}
	}
	cout << "Prediction Accuracy : " << (long double)correctPrediction / (long double)80 << endl;
}
