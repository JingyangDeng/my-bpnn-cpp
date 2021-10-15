#include "data_generator.h"

inline bool DataGenerator::is_file_exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

double DataGenerator::func(double x, double y) {
    return 1. / (1. + sqrt(x * x + y * y));
}

void DataGenerator::gen_data() {
    std::ofstream ofile(path);
    for (int i = 0; i < TRAIN_SIZE + TEST_SIZE; i++) {
        double x = (rand() % RMAX) / (double)RMAX;
        double y = (rand() % RMAX) / (double)RMAX;
        double z = func(x, y);
        ofile << std::to_string(x) << "\t" << std::to_string(y) << "\t" << std::to_string(z) << std::endl;
    }
}

void DataGenerator::get_data() {
    if (!is_file_exists(path)) {
        gen_data();
    }
    std::ifstream ifile(path);
    train_input.resize(2, std::vector<double>(TRAIN_SIZE));
    train_target.resize(1, std::vector<double>(TRAIN_SIZE));
    test_input.resize(2, std::vector<double>(TEST_SIZE));
    test_target.resize(1, std::vector<double>(TEST_SIZE));
    for (int i = 0; i < TRAIN_SIZE; i++) {
        ifile >> train_input[0][i] >> train_input[1][i] >> train_target[0][i];
    }
    for (int i = 0; i < TEST_SIZE; i++) {
        ifile >> test_input[0][i] >> test_input[1][i] >> test_target[0][i];
    }
}

DataGenerator::DataGenerator(std::string path) {
    this->path = path;
    get_data();
}
