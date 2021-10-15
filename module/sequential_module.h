#ifndef _SEQUENTIAL_MODULE_H
#define _SEQUENTIAL_MODULE_H

#include <vector>

#include "../autograd/abstract/basic_function.h"
#include "abstract/module.h"

class SequentialModule : public Module {
protected:
    int n_layers;
    std::vector<BasicFunction*> sequential;

public:
    ~SequentialModule();
    void set_target(const std::vector<std::vector<double>>& target);
    void forward(const std::vector<std::vector<double>>& input);
    void backward(const std::vector<std::vector<double>>& nothing);
    double grad_norm();
    double get_loss();
    void step(double lr);
    void show_params();
};

#endif
