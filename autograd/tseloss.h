#ifndef _TSELOSS_H
#define _TSELOSS_H

#include <vector>

#include "abstract/basic_function.h"
#include "abstract/loss_function.h"

class TSELoss : public LossFunction, public BasicFunction {
private:
    std::vector<std::vector<double>> target;  // [m, b]
    std::vector<std::vector<double>> input;   // [m, b]

public:
    void set_target(const std::vector<std::vector<double>>& target);
    double get_loss();
    void forward(const std::vector<std::vector<double>>& input);
    void backward(const std::vector<std::vector<double>>& grad_output);
};

#endif
