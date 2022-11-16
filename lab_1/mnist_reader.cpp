#include "mnist_reader.hpp"
#include <iostream>

Mnist::Mnist(int samples_amount, std::string images_filepath, std::string labels_filepath) {
    std::cout << "opened mnist";

    this->samples_amount = samples_amount;
    this->count = -1;
    
    this->image_fstream.open(images_filepath.c_str(), std::ios::in | std::ios::binary); // Binary image file
    this->label_fstream.open(labels_filepath.c_str(), std::ios::in | std::ios::binary ); // Binary label file
    
    //reading headers of files
    char number;
    for (int i = 0; i < 16; i++) {
        this->image_fstream.read(&number, sizeof(char));
	}
    for (int i = 0; i < 8; i++) {
        this->label_fstream.read(&number, sizeof(char));
	}
    curr_image = new int[this->image_size];
    curr_label = 0;
    curr_expected = new float[this->expected_size];
    for (int i = 0; i < this->expected_size; i++) {
        curr_expected[i] = 0.0;
    }
}

Mnist::~Mnist() {
    delete [] curr_image;
    delete [] curr_expected;
    std::cout << "closed mnist" << std::endl;
}

void Mnist::print_sample() {
    if (count < 0) {
        std::cout << "Read sample first";
        return;
    }
    std::cout << "Image:" << std::endl;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << curr_image[i * width + j];
        }
        std::cout << std::endl;
	}
    std::cout << "Label: " << (int)(curr_label) << std::endl;
}

void Mnist::get_next_sample() {
    if (!has_next()) {
        std::cout << "no samples found" << std::endl;
        return;
    }
    // reading image
    char number;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image_fstream.read(&number, sizeof(char));
            curr_image[i * width + j] = number == 0 ? 0 : 1;
        }
	}
	// reading label
    curr_expected[curr_label] = 0.0;
    label_fstream.read(&curr_label, sizeof(char));
    curr_expected[curr_label] = 1.0;
    count++;
}

bool Mnist::has_next() {
    return (this->count + 1) < this->samples_amount;
}

