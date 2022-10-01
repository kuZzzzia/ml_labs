#include "neural_network.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>

using namespace std;

vector<Scalar> read_bmp_file_digit(const string& filename) {
	ifstream stream(filename, std::ios::in | std::ios::binary);
	vector<uint8_t> contents((istreambuf_iterator<char>(stream)), istreambuf_iterator<char>());

	vector<Scalar> res(24);
	int index = 0;
	for (int i = 5; i >= 0; i--) {
		for (int j = 0; j < 4; j++) {
			res[i * 4 + j] = (*(contents.end() - 24 + index++) < 128 ? Scalar(0.2) : Scalar(0.8));
		}
	}

	return res;
}


void get_samples(int &batch, vector<RowVector*>& data_in, vector<RowVector*>& data_out)
{
    data_in.clear();
    data_out.clear();


    for (int i = 0; i <= 9; i++) {
		string filename = "digits/";
		filename += '0' + i;
		filename += "_";
		filename += '0' + batch;
		filename += ".bmp";
		auto parsed_vec = read_bmp_file_digit(filename);
		int cols = parsed_vec.size();
		data_in.push_back(new RowVector(1, cols));
		for (int k = 0; k < 24; k++) {
			data_in.back()->coeffRef(k) = Scalar(parsed_vec[k]);
		}
		data_out.push_back(new RowVector(1, 10));
		for (int k = 0; k < 10; k++) {
			if (k == i) {
				data_out.back()->coeffRef(k) = Scalar(1.0);
			} else {
				data_out.back()->coeffRef(k) = Scalar(0.0);
			}
		}
	}
}

 
 

int main()
{
    NeuralNetwork n({ 24, 10, 10}, 0.04);
    vector<RowVector*> in_dat, out_dat;
	for (int e = 0; e < 15; e++) {
		cout << endl << "Epoch " << e << endl;
		float acc = 0;
		for (int i = 0; i < 10; i++) {
			get_samples(i, in_dat, out_dat);
			cout << "Batch " << i << endl; 
			acc += n.train(in_dat, out_dat);
		}
		cout << "Accuracy: " << acc /100 << endl;  
	}
    return 0;
}