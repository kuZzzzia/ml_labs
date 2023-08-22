#include "mnist_reader.hpp"
#include <iostream>

Mnist::Mnist(int samples_amount, std::string images_filepath, std::string labels_filepath) {
    std::cout << "opened mnist" << std::endl;

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
    //alloc mem
    images = new int*[this->samples_amount];
    labels = new char[this->samples_amount];
    expecteds = new int*[this->samples_amount];
    for (int i = 0; i < this->samples_amount; i++) {
        images[i] = new int[this->image_size];
        expecteds[i] = new int[this->expected_size];
        for (int j = 0; j < this->expected_size; j++) {
            expecteds[i][j] = 0;
        }
    }

    for (int k = 0; k < samples_amount; k++) {
        // reading image
        char number;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                image_fstream.read(&number, sizeof(char));
                images[k][i * width + j] = number == 0 ? 0 : 1;
            }
        }
        // reading label
        label_fstream.read(&number, sizeof(char));
        labels[k] = number;
        expecteds[k][number] = 1;
    }
    std::cout << "closed mnist" << std::endl;
}

const uint Mnist::expected_size;
const uint Mnist::image_size;

Mnist::~Mnist() {
    for (int i = 0; i < samples_amount; i++) {
        delete [] images[i];
        delete [] expecteds[i];
    }
    delete [] images;
    delete [] labels;
    delete [] expecteds;
}

void Mnist::print_sample(int num) {
    if (count < 0) {
        std::cout << "Read sample first";
        return;
    }
    std::cout << "Image:" << std::endl;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << images[num][i * width + j];
        }
        std::cout << std::endl;
	}
    std::cout << "Label: " << (int)(labels[num]) << std::endl;
    std::cout << "Expected: ["; for (int i = 0; i < expected_size - 1; i++) { std::cout << expecteds[num][i] << ","; } 
    std::cout << expecteds[num][expected_size - 1] << "]" << std::endl;
}

int* Mnist::get_image() {
    return images[count];
}

int* Mnist::get_expected() {
    return expecteds[count];
}

int Mnist::get_label() {
    return labels[count];
}

void Mnist::next() {
    if (!has_next()) {
        std::cout << "no samples found" << std::endl;
        return;
    }
    count++;
}

bool Mnist::has_next() {
    return this->count + 1 < this->samples_amount;
}

void Mnist::reset_counter() {
    count = -1;
}

void Mnist::get_batches(
    const int &batch_size, std::vector<std::vector<RowVector*>>& data_in, 
    std::vector<std::vector<RowVector*>>& data_out, std::vector<std::vector<int>> &res) {

    data_in.clear();
    data_out.clear();
	res.clear();

	while(has_next()) {
		std::vector<RowVector*> in_buff, out_buff;
		std::vector<int> res_buff;
		for (int i = 0; i < batch_size; i++) {
			if (has_next()) {
				next();
				in_buff.push_back(new RowVector(1, image_size));
				int* image = get_image();
				for (int k = 0; k < image_size; k++) {
					in_buff.back()->coeffRef(k) = Scalar(image[k]);
				}
				res_buff.push_back(get_label());
				out_buff.push_back(new RowVector(1, expected_size));
				int* expected = get_expected();
				for (int k = 0; k < expected_size; k++) {
					out_buff.back()->coeffRef(k) = Scalar(expected[k]);
				}
			}
		}
		data_in.push_back(in_buff);
		data_out.push_back(out_buff);
        res.push_back(res_buff);
	}
} 
