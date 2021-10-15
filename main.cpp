#include <ctime>
#include <iostream>

#include "model/bpnn.h"
#include "model/bprelunn.h"
#include "utils/data_generator.h"
#define LR 3e-3  // LR = 1e-4 when using ReLU
#define EPS 1e-2

int main() {
    srand(time(NULL));
    DataGenerator* g = new DataGenerator("data.txt");
    std::vector<int> neurons = {2, 4, 1};

    std::cout << "architecture of network: ";
    for (int i = 0; i < (int)neurons.size(); i++)
        std::cout << neurons[i] << " ";
    std::cout << std::endl << "learning rate = " << LR << " eps = " << EPS << std::endl << std::endl;

    Model* model = new BPNN(neurons);
    // Model* model = new BPReLUNN(neurons);

    std::cout << "initial params:" << std::endl;
    model->show_params();

    model->train(g->train_input, g->train_target, EPS, LR);
    model->test(g->test_input, g->test_target);

    std::cout << std::endl << "final params:" << std::endl;
    model->show_params();

    delete g;
    delete model;
    return 0;
}
