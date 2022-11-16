#include <string>
#include <fstream>

class Mnist {
public:
    static const uint width = 28;
    static const uint height = 28;
    const static uint image_size = width * height;
    const static uint expected_size = 10; // 10 numbers from 0 to 9 
    
    int* curr_image;
    char curr_label;
    float* curr_expected;
    std::string images_filepath;
    std::string labels_filepath;

    Mnist(int samples_amount, std::string images_filepath, std::string labels_filepath);
    virtual ~Mnist();

    void get_next_sample();
    bool has_next();
    void print_sample();
private: 
    int samples_amount; // 60 000 for train, 10 000 for test
    int count;
    std::ifstream image_fstream;
    std::ifstream label_fstream;
};