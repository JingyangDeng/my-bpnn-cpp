#include "relu.h"

#include "../propagation/propagation.h"

void ReLU::forward(const std::vector<std::vector<double>>& input) {
    this->input = input;
    relu(input, output);
}

void ReLU::backward(const std::vector<std::vector<double>>& grad_output) {
    relu_back(grad_output, input, grad_input);
}
