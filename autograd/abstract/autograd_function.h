#ifndef _AUTOGRAD_FUNCTION_H
#define _AUTOGRAD_FUNCTION_H

#include <vector>

class AutoGradFunction {
public:
    virtual void forward(const std::vector<std::vector<double>>&) = 0;
    virtual void backward(const std::vector<std::vector<double>>&) = 0;
    virtual ~AutoGradFunction();
};

#endif
