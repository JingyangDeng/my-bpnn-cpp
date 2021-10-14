#include <sys/stat.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#define TRAIN_SIZE 5000
#define TEST_SIZE 5000
#define RMAX 1001
using namespace std;

class DataGenerator {
private:
    string path;

    inline bool is_file_exists(const string& name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }

    double func(double x, double y) {
        return 1. / (1. + sqrt(x * x + y * y));
    }

    void gen_data() {
        ofstream ofile(path);
        for (int i = 0; i < TRAIN_SIZE + TEST_SIZE; i++) {
            double x = (rand() % RMAX) / (double)RMAX;
            double y = (rand() % RMAX) / (double)RMAX;
            double z = func(x, y);
            ofile << to_string(x) << "\t" << to_string(y) << "\t" << to_string(z) << endl;
        }
    }

    void get_data() {
        if (!is_file_exists(path)) {
            gen_data();
        }
        ifstream ifile(path);
        train_input.resize(2, vector<double>(TRAIN_SIZE));
        train_target.resize(1, vector<double>(TRAIN_SIZE));
        test_input.resize(2, vector<double>(TEST_SIZE));
        test_target.resize(1, vector<double>(TEST_SIZE));
        for (int i = 0; i < TRAIN_SIZE; i++) {
            ifile >> train_input[0][i] >> train_input[1][i] >> train_target[0][i];
        }
        for (int i = 0; i < TEST_SIZE; i++) {
            ifile >> test_input[0][i] >> test_input[1][i] >> test_target[0][i];
        }
    }

public:
    vector<vector<double>> train_input;
    vector<vector<double>> train_target;
    vector<vector<double>> test_input;
    vector<vector<double>> test_target;

    DataGenerator(string path) {
        this->path = path;
        get_data();
    }
};
