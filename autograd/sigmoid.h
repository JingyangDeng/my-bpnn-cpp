#ifndef _SIGMOID_H
#define _SIGMOID_H

#include <vector>

#include "abstract/basic_function.h"

class Sigmoid : public BasicFunction {
public:
    void forward(const std::vector<std::vector<double>>& input);
    void backward(const std::vector<std::vector<double>>& grad_output);
};

#endif
