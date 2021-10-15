#include "basic_function.h"

const std::vector<std::vector<double>>& BasicFunction::get_output() {
    return output;
}

const std::vector<std::vector<double>>& BasicFunction::get_grad_input() {
    return grad_input;
}

BasicFunction::~BasicFunction() {
}
