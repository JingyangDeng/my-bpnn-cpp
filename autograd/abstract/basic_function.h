#ifndef _BASIC_FUNCTION_H
#define _BASIC_FUNCTION_H

#include <vector>

#include "autograd_function.h"

class BasicFunction : public AutoGradFunction {
protected:
    std::vector<std::vector<double>> output;
    std::vector<std::vector<double>> grad_input;

public:
    const std::vector<std::vector<double>>& get_output();
    const std::vector<std::vector<double>>& get_grad_input();
    virtual ~BasicFunction();
};

#endif
