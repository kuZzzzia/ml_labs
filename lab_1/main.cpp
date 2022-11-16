#include "mnist_reader.hpp"
#include "neural_network.hpp"

#include <vector>
#include <stdlib.h>


void get_samples(Mnist &samples, const int &batch_size, std::vector<RowVector*>& data_in, std::vector<RowVector*>& data_out) {
    data_in.clear();
    data_out.clear();

    for (int i = 0; i < batch_size; i++) {
		if (samples.has_next()) {
			samples.get_next_sample();

			data_in.push_back(new RowVector(1, samples.image_size));
			for (int k = 0; k < samples.image_size; k++) {
				data_in.back()->coeffRef(k) = Scalar(samples.curr_image[k]);
			}

			data_out.push_back(new RowVector(1, samples.expected_size));
			for (int k = 0; k < samples.expected_size; k++) {
				data_out.back()->coeffRef(k) = Scalar(samples.curr_expected[k]);
			}
		}
	}
} 

using namespace std;

int main() {
	const int train_samples_amount = 60000;
	const string train_images_filepath = "../../mnist/train-images.idx3-ubyte";
	const string train_labels_filepath = "../../mnist/train-labels.idx1-ubyte";

	const int test_samples_amount = 10000;
	const string test_images_filepath = "../../mnist/train-images.idx3-ubyte";
	const string test_labels_filepath = "../../mnist/train-images.idx3-ubyte";
	
	const int epoch_amount = 3;
	const int batch_size = 5000;
	const double learning_rate = 0.7;

	NeuralNetwork n({ Mnist::image_size, 30, Mnist::expected_size }, learning_rate);
    vector<RowVector*> in_dat, out_dat;
	for (int e = 0; e < epoch_amount; e++) {
		cout << endl << "Epoch " << e + 1 << endl;
		Mnist train_samples = Mnist(train_samples_amount, train_images_filepath, train_labels_filepath);
		float acc = 0;
		
		cout << "Batches: " << endl;
		for (int i = 0; i < train_samples_amount / batch_size; i++) {
			cout << i + 1 << endl;
			get_samples(train_samples, batch_size, in_dat, out_dat);
			acc += n.train(in_dat, out_dat);
		}
		cout << endl << "Accuracy: " << acc /train_samples_amount << endl;  
	}


    return 0;
}