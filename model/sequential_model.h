#ifndef _SEQUENTIAL_MODEL_H
#define _SEQUENTIAL_MODEL_H

#include <vector>

#include "../module/sequential_module.h"
#include "abstract/model.h"

class SequentialModel : public Model {
protected:
    SequentialModule* network;

public:
    void train(const std::vector<std::vector<double>>& train_input,
               const std::vector<std::vector<double>>& train_target, double eps, double lr);
    void test(const std::vector<std::vector<double>>& test_input, const std::vector<std::vector<double>>& test_target);
    void show_params();
    bool stop(double loss, double grad, double eps);
    ~SequentialModel();
};

#endif
