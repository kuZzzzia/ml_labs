#include "neural_network.hpp"
#include "mnist_reader.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>

using namespace std;


void get_samples(Mnist &samples, const int &batch_size, vector<RowVector*>& data_in, vector<RowVector*>& data_out) {
    data_in.clear();
    data_out.clear();

    for (int i = 0; i < batch_size; i++) {
		if (samples.has_next()) {
			samples.get_next_sample();
			samples.print_sample();

			data_in.push_back(new RowVector(1, samples.image_size));
			for (int k = 0; k < samples.image_size; k++) {
				data_in.back()->coeffRef(k) = Scalar(samples.curr_image[k]);
			}

			data_out.push_back(new RowVector(1, samples.expected_size));
			for (int k = 0; k < samples.expected_size; k++) {
				data_in.back()->coeffRef(k) = Scalar(samples.curr_expected[k]);
			}
		}
	}
} 

int main() {
	const int train_samples_amount = 60000;
	const int epoch_amount = 1;
	const int batch_size = 1000;
	const double learning_rate = 0.04;
	const int epoch_amount = 2;

	Mnist samples = Mnist(train_samples_amount, "../../mnist/train-images.idx3-ubyte", "../../mnist/train-labels.idx1-ubyte");
	NeuralNetwork n({ samples.image_size, samples.expected_size }, learning_rate);
    vector<RowVector*> in_dat, out_dat;
	for (int e = 0; e < epoch_amount; e++) {
		cout << endl << "Epoch " << e + 1 << endl;
		float acc = 0;
		
		for (int i = 0; i < train_samples_amount / batch_size; i++) {
			get_samples(samples, batch_size, in_dat, out_dat);
			acc += n.train(in_dat, out_dat);
		}
		cout << "Accuracy: " << acc / 100 << endl;  
	}
    return 0;
}