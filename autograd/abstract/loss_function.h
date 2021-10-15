#ifndef _LOSS_FUNCTION_H
#define _LOSS_FUNCTION_H

#include <vector>

class LossFunction {
public:
    virtual void set_target(const std::vector<std::vector<double>>&) = 0;
    virtual double get_loss() = 0;
    virtual ~LossFunction();
};

#endif
