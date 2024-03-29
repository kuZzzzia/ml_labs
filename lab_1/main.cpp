#include "mnist_reader.hpp"
#include "neural_network.hpp"

#include <vector>
#include <stdlib.h>
#include <chrono>

using namespace std;

int main() {
	const int train_samples_amount = 60000;
	const string train_images_filepath = "../../mnist/train-images.idx3-ubyte";
	const string train_labels_filepath = "../../mnist/train-labels.idx1-ubyte";

	const int test_samples_amount = 10000;
	const string test_images_filepath = "../../mnist/t10k-images.idx3-ubyte";
	const string test_labels_filepath = "../../mnist/t10k-labels.idx1-ubyte";
	
	const int epoch_amount = 5;
	const int batch_size = 5000;
	const double learning_rate = 0.5;

	NeuralNetwork n({ Mnist::image_size, 16,  Mnist::expected_size }, learning_rate);
	Mnist train_samples = Mnist(train_samples_amount, train_images_filepath, train_labels_filepath);
	Mnist test_samples = Mnist(test_samples_amount, test_images_filepath, test_labels_filepath);
    vector<vector<RowVector*>> in_dat, out_dat, in_test, out_test;
	vector<vector<int>> train_res, test_res;
	train_samples.get_batches(batch_size, in_dat, out_dat, train_res);
	test_samples.get_batches(test_samples_amount, in_test, out_test, test_res);

	for (int e = 0; e < epoch_amount; e++) {
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		cout << "Epoch " << e + 1 << "/" << epoch_amount <<  endl;
	
		for (int i = 0; i < train_samples_amount / batch_size; i++) {
			n.train(in_dat[i], out_dat[i], train_res[i]);
		}
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		std::cout << "\ttime elapsed: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
		n.test(in_test[0], out_test[0], test_res[0]);

		train_samples.reset_counter();
		test_samples.reset_counter();
	}
    return 0;
}