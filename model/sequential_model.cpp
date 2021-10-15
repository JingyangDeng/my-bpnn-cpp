#include "sequential_model.h"

#include <iostream>

void SequentialModel::train(const std::vector<std::vector<double>>& train_input,
                            const std::vector<std::vector<double>>& train_target, double eps, double lr) {
    int iter = 0;
    double loss, grad;
    while (1) {
        network->set_target(train_target);
        network->forward(train_input);
        loss = network->get_loss();
        network->backward(std::vector<std::vector<double>>(0));
        grad = network->grad_norm();
        if (iter % 1000 == 0) {
            std::cout << "iter = " << iter << " loss = " << loss << " grad = " << grad << std::endl;
        }
        if (stop(loss, grad, eps))
            break;
        network->step(lr);
        iter++;
    }
}

void SequentialModel::test(const std::vector<std::vector<double>>& test_input,
                           const std::vector<std::vector<double>>& test_target) {
    network->set_target(test_target);
    network->forward(test_input);
    double loss = network->get_loss();
    std::cout << "test: tseloss = " << loss << std::endl;
}

void SequentialModel::show_params() {
    network->show_params();
}

bool SequentialModel::stop(double loss, double grad, double eps) {
    return loss < eps || grad < eps;
}

SequentialModel::~SequentialModel() {
    delete network;
}
