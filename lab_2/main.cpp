#include "mnist_reader.hpp"
#include "neural_network.hpp"
#include <fstream>

#include <vector>
#include <stdlib.h>
#include <chrono>

using namespace std;

const int train_samples_amount = 60000;
const string train_images_filepath = "../../mnist/train-images.idx3-ubyte";
const string train_labels_filepath = "../../mnist/train-labels.idx1-ubyte";

const int test_samples_amount = 10000;
const string test_images_filepath = "../../mnist/t10k-images.idx3-ubyte";
const string test_labels_filepath = "../../mnist/t10k-labels.idx1-ubyte";

const int epoch_amount = 5;
const int batch_size = 1000;
const double learning_rate = 0.4;

vector<vector<RowVector*>> in_dat, out_dat, in_test, out_test;
vector<vector<int>> train_res, test_res;


void trainWithOptimizer(ofstream &myfile, Optimizer o, Mnist &train_samples, Mnist &test_samples) {

	NeuralNetwork n({ Mnist::image_size, 16,  Mnist::expected_size }, learning_rate, o);
	for (int e = 0; e < epoch_amount; e++) {
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		cout << "Epoch " << e + 1 << "/" << epoch_amount <<  endl;
	
		for (int i = 0; i < train_samples_amount / batch_size; i++) {
			n.train(in_dat[i], out_dat[i], train_res[i]);
		}
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		myfile << o << "," << e+1 << ","; 
		myfile << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << ",";
		n.test(myfile, in_test[0], out_test[0], test_res[0]);

		train_samples.reset_counter();
		test_samples.reset_counter();
	}
}

int main() {
	Mnist train_samples = Mnist(train_samples_amount, train_images_filepath, train_labels_filepath);
	Mnist test_samples = Mnist(test_samples_amount, test_images_filepath, test_labels_filepath);
    
	train_samples.get_batches(batch_size, in_dat, out_dat, train_res);
	test_samples.get_batches(test_samples_amount, in_test, out_test, test_res);

	ofstream myfile;
  	myfile.open ("res/res.csv");
  	myfile << "Optimizer,Epoch,Time,Accuracy,Loss\n";

	// trainWithOptimizer(myfile, SGD, train_samples, test_samples);
	// trainWithOptimizer(myfile, Momentum, train_samples, test_samples);
	// trainWithOptimizer(myfile, AdaGrad, train_samples, test_samples);
	// trainWithOptimizer(myfile, RMSprop, train_samples, test_samples);
	trainWithOptimizer(myfile, Adam, train_samples, test_samples);

	myfile.close();

    return 0;
}