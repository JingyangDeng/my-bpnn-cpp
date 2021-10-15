#ifndef _LINEAR_H
#define _LINEAR_H

#include <vector>

#include "abstract/basic_function.h"
#include "abstract/parametrized_function.h"
#define ALPHA 5.
#define MAX 1000

class Linear : public ParametrizedFunction, public BasicFunction {
private:
    std::vector<std::vector<double>> weight;
    std::vector<std::vector<double>> input;        // [n_(k-1), b]
    std::vector<std::vector<double>> grad_weight;  // [nk, n_(k-1) + 1]
    void reset_params();

public:
    Linear(int in_features, int out_features);
    void forward(const std::vector<std::vector<double>>& input);
    double grad_norm();
    void show_params();
    void backward(const std::vector<std::vector<double>>& grad_output);
    void step(double lr);
};

#endif
