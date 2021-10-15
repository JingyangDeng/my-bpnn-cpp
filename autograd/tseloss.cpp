#include "tseloss.h"

#include "../propagation/propagation.h"

void TSELoss::set_target(const std::vector<std::vector<double>>& target) {
    this->target = target;
}

double TSELoss::get_loss() {
    return output[0][0];
}

void TSELoss::forward(const std::vector<std::vector<double>>& input) {
    this->input = input;
    tseloss(input, target, output);
}

void TSELoss::backward(const std::vector<std::vector<double>>& grad_output) {
    tseloss_back(input, target, grad_input);
}
