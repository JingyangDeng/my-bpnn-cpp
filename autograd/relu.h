#ifndef _RELU_H
#define _RELU_H

#include <vector>

#include "abstract/basic_function.h"

class ReLU : public BasicFunction {
private:
    std::vector<std::vector<double>> input;

public:
    void forward(const std::vector<std::vector<double>>& input);
    void backward(const std::vector<std::vector<double>>& grad_output);
};

#endif
