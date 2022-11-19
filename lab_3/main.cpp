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


void trainWithOptimizer(ofstream &myfile, int layers, Scalar l1, Scalar l2, Mnist &train_samples, Mnist &test_samples) {
	vector<uint> topology;
	for (int i = 0; i < layers; i++) {
		topology.push_back(10);
	}
	topology.push_back(Mnist::expected_size);
	topology.insert(topology.begin(), Mnist::image_size);

	NeuralNetwork n(topology, learning_rate, l1, l2, SGD);
	for (int e = 0; e < epoch_amount; e++) {
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		cout << "Epoch " << e + 1 << "/" << epoch_amount <<  endl;
	
		for (int i = 0; i < train_samples_amount / batch_size; i++) {
			n.train(in_dat[i], out_dat[i], train_res[i]);
		}
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		if (l1 != 0 && l2 != 0) {
			myfile << "L1+L2,";
		} else if (l2 != 0) {
			myfile << "L1,";
		} else if (l1 != 0) {
			myfile << "L2,";
		} else {
			myfile << layers;
		}
		myfile << e+1 << ","; 
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
  	myfile.open ("res/res1.csv");
  	myfile << "Hidden,Epoch,Time,Accuracy,Loss\n";

	trainWithOptimizer(myfile, 0, 0, 0, train_samples, test_samples);
	trainWithOptimizer(myfile, 1, 0, 0, train_samples, test_samples);
	trainWithOptimizer(myfile, 2, 0, 0, train_samples, test_samples);
	trainWithOptimizer(myfile, 3, 0, 0, train_samples, test_samples);
	myfile.close();

	myfile.open ("res/res2.csv");
  	myfile << "Regularisation,Epoch,Time,Accuracy,Loss\n";
	trainWithOptimizer(myfile, 1, 0, 0, train_samples, test_samples);
	trainWithOptimizer(myfile, 1, 0.001, 0, train_samples, test_samples);
	trainWithOptimizer(myfile, 1, 0, 0.001, train_samples, test_samples);
	trainWithOptimizer(myfile, 1, 0.005, 0.005, train_samples, test_samples);

	myfile.close();

    return 0;
}