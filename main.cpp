#include <iostream>
#include <vector>

#include "data_generator.cpp"
#include "model.cpp"
#define LR 3e-3  // LR = 1e-4 when using ReLU
#define EPS 1e-2
using namespace std;

int main() {
    srand(time(NULL));
    DataGenerator* g = new DataGenerator("data.txt");
    vector<int> neurons = {2, 4, 1};

    cout << "architecture of network: ";
    for (int i = 0; i < (int)neurons.size(); i++)
        cout << neurons[i] << " ";
    cout << endl << "learning rate = " << LR << " eps = " << EPS << endl << endl;

    Model* model = new BPNN(neurons);
    // Model* model = new BPReLUNN(neurons);

    cout << "initial params:" << endl;
    model->show_params();

    model->train(g->train_input, g->train_target, EPS, LR);
    model->test(g->test_input, g->test_target);

    cout << endl << "final params:" << endl;
    model->show_params();

    delete g;
    delete model;
    return 0;
}
