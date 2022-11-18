#include <string>
#include <fstream>
#include <vector>

#ifndef import
#include <eigen3/Eigen/Eigen>
typedef float Scalar;
typedef Eigen::RowVectorXd RowVector;
#endif

class Mnist {
public:
    static const uint width = 28;
    static const uint height = 28;
    const static uint image_size = width * height;
    const static uint expected_size = 10; // 10 numbers from 0 to 9 
    
    int** images;
    char* labels;
    int** expecteds;
    std::string images_filepath;
    std::string labels_filepath;

    Mnist(int samples_amount, std::string images_filepath, std::string labels_filepath);
    virtual ~Mnist();

    int* get_image();
    int* get_expected();
    int get_label();
    bool has_next();
    void next();
    void print_sample(int);
    void reset_counter();
    void get_batches(
        const int &batch_size, 
        std::vector<std::vector<RowVector*>>& data_in, 
        std::vector<std::vector<RowVector*>>& data_out, 
        std::vector<std::vector<int>> &res
    );
private: 
    int samples_amount; // 60 000 for train, 10 000 for test
    int count;
    std::ifstream image_fstream;
    std::ifstream label_fstream;
};