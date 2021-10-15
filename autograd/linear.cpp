#include "linear.h"

#include <iostream>

#include "../propagation/propagation.h"

void Linear::reset_params() {
    for (int i = 0; i < (int)weight.size(); i++) {
        for (int j = 0; j < (int)weight[0].size(); j++) {
            weight[i][j] = ALPHA * ((rand() % MAX) / (double)MAX - 0.5);
        }
    }
}

Linear::Linear(int in_features, int out_features) {
    weight.resize(out_features, std::vector<double>(in_features + 1));
    reset_params();
}

void Linear::forward(const std::vector<std::vector<double>>& input) {
    this->input = input;
    linear(weight, input, output);
}

double Linear::grad_norm() {
    return norm(grad_weight);
}

void Linear::show_params() {
    for (int i = 0; i < (int)weight.size(); i++) {
        for (int j = 0; j < (int)weight[0].size(); j++) {
            std::cout << weight[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void Linear::backward(const std::vector<std::vector<double>>& grad_output) {
    linear_back(grad_output, input, weight, grad_input, grad_weight);
}

void Linear::step(double lr) {
    update(weight, grad_weight, lr);
}
