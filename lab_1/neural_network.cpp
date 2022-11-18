#include "neural_network.hpp"

Scalar sigmoid(Scalar x) {
    return 1.0 / (1.0 + exp(-x));
}
 
Scalar sigmoidDerivative(Scalar x) {
    return sigmoid(x)*(1-sigmoid(x));
}

NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate)
{
    this->topology = topology;
    this->learningRate = learningRate;

    for (uint i = 0; i < topology.size(); i++) {
        // initialize neuron layers
        neuronLayers.push_back(new RowVector(topology[i]));
 
        // initialize cache and delta vectors
        cacheLayers.push_back(new RowVector(topology[i]));
        deltas.push_back(new RowVector(topology[i]));
 
        // initialize weights matrix
        if (i > 0) {
            weights.push_back(new Matrix(topology[i - 1], topology[i]));
            weights.back()->setRandom();
        }
    }
};


void NeuralNetwork::propagateForward(RowVector& input)
{
    // set the input to input layer
    // block returns a part of the given vector or matrix
    // block takes 4 arguments : startRow, startCol, blockRows, blockCols
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size()) = input;
 
    // propagate the data forward and then
    for (uint i = 1; i < topology.size(); i++) {
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
        (*cacheLayers[i]) = (*neuronLayers[i]);
        for (uint j = 0; j < topology[i]; j++) {
            neuronLayers[i]->coeffRef(j) = sigmoid(neuronLayers[i]->coeffRef(j));
        }
    }
}


void NeuralNetwork::calcErrors(RowVector& output)
{
    // calculate the errors made by neurons of last layer
    (*deltas.back()) = output - (*neuronLayers.back());
 
    // error calculation of hidden layers is different
    // we will begin by the last hidden layer
    // and we will continue till the first hidden layer
    for (uint i = topology.size() - 2; i > 0; i--) {
        (*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
    }
}

void NeuralNetwork::updateWeights()
{
    // topology.size()-1 = weights.size()
    for (uint i = 0; i < topology.size() - 1; i++) {
        for (uint c = 0; c < weights[i]->cols(); c++) {
            for (uint r = 0; r < weights[i]->rows(); r++) {
                weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * sigmoidDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
            }
        }
    }
}

void NeuralNetwork::propagateBackward(RowVector& output)
{
    calcErrors(output);
    updateWeights();
}

int NeuralNetwork::get_res_from_output() {
    RowVector* v = neuronLayers.back();
    double max = v->coeff(0);
    int arg = 0; 
    for (int j = 1; j < 10; j++) {
        if (v->coeff(j) > max) {
           max = v->coeff(j);
           arg = j;
        }
    }
    return arg;
}

Scalar NeuralNetwork::calc_cost(RowVector* x) {
    double err = 0;
    RowVector *res = neuronLayers.back();
    for (int i = 0; i < x->size(); i++) {
        err += (x->coeff(i) - res->coeff(i))*(x->coeff(i) - res->coeff(i));
    }
    return err / 2;
}



void NeuralNetwork::train(std::vector<RowVector*> &input, std::vector<RowVector*> &output, std::vector<int> &res)
{
    for (uint i = 0; i < input.size(); i++) {
        propagateForward(*input[i]);
        propagateBackward(*output[i]);
    }
}

void NeuralNetwork::test(std::vector<RowVector*> &input, std::vector<RowVector*> &output, std::vector<int> &res) {
    Scalar accuracy = 0;
    Scalar loss = 0;
    for (int i = 0; i < input.size(); i++) {
        propagateForward(*input[i]);
        int number_guessed = get_res_from_output();
        if (res[i] == number_guessed) {
            accuracy++;
        }
        loss += calc_cost(output[i]);
    }
    std::cout << "accuracy: " << accuracy / res.size() << " , loss: " << loss / res.size() << std::endl << std::endl;
}