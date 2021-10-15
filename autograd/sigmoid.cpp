#include "sigmoid.h"

#include "../propagation/propagation.h"

void Sigmoid::forward(const std::vector<std::vector<double>>& input) {
    sigmoid(input, output);
}

void Sigmoid::backward(const std::vector<std::vector<double>>& grad_output) {
    sigmoid_back(grad_output, output, grad_input);
}
