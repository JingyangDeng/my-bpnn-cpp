#ifndef _DATA_GENERATOR_H
#define _DATA_GENERATOR_H

#include <sys/stat.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#define TRAIN_SIZE 5000
#define TEST_SIZE 5000
#define RMAX 1001

class DataGenerator {
private:
    std::string path;
    bool is_file_exists(const std::string& name);
    double func(double x, double y);
    void gen_data();
    void get_data();

public:
    std::vector<std::vector<double>> train_input;
    std::vector<std::vector<double>> train_target;
    std::vector<std::vector<double>> test_input;
    std::vector<std::vector<double>> test_target;
    DataGenerator(std::string path);
};

#endif
